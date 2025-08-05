"""QLoRA 학습 루프 구현"""

from itertools import cycle
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


class QLoRATrainer:
    """QLoRA 학습을 수행하는 트레이너"""

    def __init__(
        self,
        model,
        tokenizer,
        dataset,
        output_dir: Path,
        checkpoint_dir: Path,
        log_dir: Path,
        max_steps: int,
        batch_size: int,
        lr: float,
        save_steps: int,
        log_steps: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.lr = lr
        self.save_steps = save_steps
        self.log_steps = log_steps

        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.is_main = self.rank == 0

    def _setup(self):
        sampler = DistributedSampler(self.dataset) if self.world_size > 1 else None
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
        )

        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scaler = GradScaler()
        if self.is_main:
            self.writer = SummaryWriter(self.log_dir)

    def _save_checkpoint(self, step: int):
        if not self.is_main:
            return
        ckpt_path = self.checkpoint_dir / f"step_{step}.pt"
        state = {
            "model": self.model.state_dict() if not isinstance(self.model, DDP) else self.model.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
        }
        torch.save(state, ckpt_path)

    def train(self) -> bool:
        self._setup()
        self.model.train()

        step = 0
        data_iter = cycle(self.dataloader)

        while step < self.max_steps:
            batch: Dict[str, torch.Tensor] = next(data_iter)
            batch = {k: v.cuda() for k, v in batch.items()}

            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = self.model(**batch)
                loss = outputs.loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if self.is_main and step % self.log_steps == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)

            if step % self.save_steps == 0:
                self._save_checkpoint(step)

            step += 1

        if self.is_main:
            self.writer.close()
        return True
