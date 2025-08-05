"""TRELLIS QLoRA 학습 매니저"""

from pathlib import Path
from typing import Optional

from model_loader import ModelLoader
from data_loader import TextDataset
from trainer import QLoRATrainer


class TRELLISQLoRAManager:
    """QLoRA 실험을 관리하는 클래스"""

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        output_dir: str,
        max_steps: int,
        batch_size: int,
        lr: float,
        save_steps: int,
        log_steps: int,
    ) -> None:
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.lr = lr
        self.save_steps = save_steps
        self.log_steps = log_steps

        self.checkpoint_dir = self.output_dir / "ckpts"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.log_dir.mkdir(exist_ok=True, parents=True)

        self.model = None
        self.tokenizer = None
        self.dataset = None

    def prepare(self) -> bool:
        """모델과 데이터셋 준비"""
        try:
            loader = ModelLoader(self.model_path)
            self.model, self.tokenizer = loader.load_model()
            self.dataset = TextDataset(self.dataset_path, self.tokenizer)
            return True
        except Exception as e:  # pragma: no cover - 단순 출력
            print(f"❌ 준비 단계 실패: {e}")
            return False

    def run_experiment(self) -> bool:
        """QLoRA 학습 실행"""
        if not self.prepare():
            return False

        trainer = QLoRATrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            output_dir=self.output_dir,
            checkpoint_dir=self.checkpoint_dir,
            log_dir=self.log_dir,
            max_steps=self.max_steps,
            batch_size=self.batch_size,
            lr=self.lr,
            save_steps=self.save_steps,
            log_steps=self.log_steps,
        )

        return trainer.train()
