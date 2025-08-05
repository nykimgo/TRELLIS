"""
TRELLIS QLoRA 트레이너

주요 기능:
- QLoRA를 이용한 TRELLIS 모델 fine-tuning
- 분산 훈련 (DDP) 지원
- Mixed Precision (FP16) 지원
- 체크포인트 저장/로드
- TensorBoard 로깅
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# QLoRA 관련 라이브러리
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

# TRELLIS 관련 임포트
try:
    from trellis.pipelines import TrellisTextTo3DPipeline
    from trellis.models import TrellisImageTokenizer, TrellisSparseStructureDecoder
except ImportError:
    print("⚠️ TRELLIS 모듈을 임포트할 수 없습니다. PYTHONPATH를 확인하세요.")

from qlora_config import QLoRAConfig
from utils.data_loader import create_dataloader
from utils.model_utils import get_trainable_parameters
from utils.optimizer import create_optimizer, create_scheduler
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.evaluation import evaluate_model


class TRELLISQLoRATrainer:
    """TRELLIS QLoRA 트레이너"""
    
    def __init__(self, config: QLoRAConfig, exp_dir: Path, logger: logging.Logger):
        """초기화"""
        self.config = config
        self.exp_dir = exp_dir
        self.logger = logger
        
        # 분산 훈련 설정
        self.setup_distributed()
        
        # 시드 설정
        self.setup_seed()
        
        # 모델 관련 변수
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # 훈련 상태
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
        # TensorBoard
        if self.is_master and config.use_tensorboard:
            self.writer = SummaryWriter(exp_dir / "tensorboard")
        else:
            self.writer = None
    
    def setup_distributed(self):
        """분산 훈련 설정"""
        if self.config.num_gpus > 1:
            if not dist.is_available():
                raise RuntimeError("분산 훈련이 지원되지 않습니다")
            
            # 환경 변수 설정
            if 'RANK' not in os.environ:
                os.environ['RANK'] = '0'
                os.environ['WORLD_SIZE'] = str(self.config.num_gpus)
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
            
            dist.init_process_group(backend='nccl')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.rank)
            self.device = torch.device(f'cuda:{self.rank}')
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.is_master = (self.rank == 0)
        
        if self.is_master:
            self.logger.info(f"🔧 분산 설정: rank={self.rank}, world_size={self.world_size}")
    
    def setup_seed(self):
        """시드 설정"""
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup_model(self):
        """모델 설정"""
        self.logger.info(f"📥 모델 로드: {self.config.model_path}")
        
        # BitsAndBytesConfig 설정
        bnb_config = BitsAndBytesConfig(**self.config.quantization_config)
        
        # TRELLIS 파이프라인 로드
        pipeline = TrellisTextTo3DPipeline.from_pretrained(
            self.config.model_path,
            quantization_config=bnb_config,
            device_map={"": self.device}
        )
        
        # QLoRA 설정할 메인 모델 선택 (일반적으로 sparse_structure_decoder)
        if hasattr(pipeline, 'sparse_structure_decoder'):
            base_model = pipeline.sparse_structure_decoder
        elif hasattr(pipeline.models, 'sparse_structure_decoder'):
            base_model = pipeline.models['sparse_structure_decoder']
        else:
            raise ValueError("적절한 base model을 찾을 수 없습니다")
        
        # LoRA 설정
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type=TaskType.FEATURE_EXTRACTION  # 3D 생성은 feature extraction으로 간주
        )
        
        # LoRA 모델 생성
        self.model = get_peft_model(base_model, lora_config)
        
        # 그래디언트 체크포인팅
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # 분산 모델로 래핑
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        
        # 파라미터 정보 출력
        trainable_params, total_params = get_trainable_parameters(self.model)
        self.logger.info(f"📊 파라미터: {trainable_params:,} / {total_params:,} "
                        f"({100 * trainable_params / total_params:.2f}%)")
        
        # 파이프라인의 모델 교체
        if hasattr(pipeline, 'sparse_structure_decoder'):
            pipeline.sparse_structure_decoder = self.model.module if isinstance(self.model, DDP) else self.model
        
        self.pipeline = pipeline
    
    def setup_data(self):
        """데이터로더 설정"""
        self.logger.info("📊 데이터로더 설정")
        
        # 데이터로더 생성
        self.train_loader, self.val_loader = create_dataloader(
            self.config, self.world_size, self.rank
        )
        
        self.logger.info(f"📊 훈련 데이터: {len(self.train_loader)} 배치")
        if self.val_loader:
            self.logger.info(f"📊 검증 데이터: {len(self.val_loader)} 배치")
    
    def setup_optimizer(self):
        """옵티마이저 및 스케줄러 설정"""
        self.logger.info("🔧 옵티마이저 설정")
        
        # 옵티마이저 생성
        self.optimizer = create_optimizer(self.model, self.config)
        
        # 스케줄러 생성
        self.scheduler = create_scheduler(self.optimizer, self.config)
        
        # Mixed Precision Scaler
        if self.config.fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.logger.info(f"🔧 옵티마이저: {self.optimizer.__class__.__name__}")
        self.logger.info(f"🔧 스케줄러: {self.scheduler.__class__.__name__}")
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """단일 훈련 스텝"""
        self.model.train()
        
        # 데이터를 GPU로 이동
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device, non_blocking=True)
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.fp16):
            outputs = self.model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return {'loss': loss.item() * self.config.gradient_accumulation_steps}
    
    def train_epoch(self):
        """에포크 훈련"""
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(self.epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        for step, batch in enumerate(self.train_loader):
            # 훈련 스텝
            metrics = self.train_step(batch)
            total_loss += metrics['loss']
            num_batches += 1
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    if self.config.fp16:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clipping
                    )
                
                # Optimizer step
                if self.config.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # 로깅
                if self.global_step % self.config.logging_steps == 0:
                    self.log_metrics({
                        'train/loss': metrics['loss'],
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/step': self.global_step
                    })
                
                # 평가
                if self.global_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self.log_metrics(eval_metrics)
                
                # 체크포인트 저장
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                
                # 최대 스텝 도달 시 종료
                if self.global_step >= self.config.max_steps:
                    return total_loss / max(num_batches, 1)
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self) -> Dict[str, float]:
        """모델 평가"""
        if not self.val_loader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 데이터를 GPU로 이동
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=self.config.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 분산 환경에서 평균 계산
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        
        metrics = {
            'eval/loss': avg_loss,
            'eval/step': self.global_step
        }
        
        # 베스트 모델 저장
        if avg_loss < self.best_metric:
            self.best_metric = avg_loss
            if self.is_master:
                self.save_best_model()
        
        return metrics
    
    def log_metrics(self, metrics: Dict[str, float]):
        """메트릭 로깅"""
        if not self.is_master:
            return
        
        # 콘솔 출력
        if 'train/loss' in metrics:
            self.logger.info(
                f"Step {self.global_step:6d} | "
                f"Loss: {metrics['train/loss']:.4f} | "
                f"LR: {metrics['train/lr']:.2e}"
            )
        
        if 'eval/loss' in metrics:
            self.logger.info(
                f"Eval Step {self.global_step:6d} | "
                f"Loss: {metrics['eval/loss']:.4f}"
            )
        
        # TensorBoard 로깅
        if self.writer:
            for key, value in metrics.items():
                if key != 'train/step' and key != 'eval/step':
                    self.writer.add_scalar(key, value, self.global_step)
    
    def save_checkpoint(self):
        """체크포인트 저장"""
        if not self.is_master:
            return
        
        checkpoint_path = self.exp_dir / "checkpoints" / f"checkpoint-{self.global_step}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # 모델 상태 저장
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_save.save_pretrained(checkpoint_path)
        
        # 훈련 상태 저장
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config.__dict__
        }
        
        if self.config.fp16:
            state['scaler'] = self.scaler.state_dict()
        
        torch.save(state, checkpoint_path / "training_state.pt")
        
        self.logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
        
        # 오래된 체크포인트 정리
        self.cleanup_checkpoints()
    
    def save_best_model(self):
        """베스트 모델 저장"""
        best_path = self.exp_dir / "best_model"
        best_path.mkdir(exist_ok=True)
        
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_save.save_pretrained(best_path)
        
        self.logger.info(f"🏆 베스트 모델 저장: {best_path} (loss: {self.best_metric:.4f})")
    
    def cleanup_checkpoints(self):
        """오래된 체크포인트 정리"""
        checkpoint_dir = self.exp_dir / "checkpoints"
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        
        if len(checkpoints) > self.config.save_total_limit:
            # 스텝 번호로 정렬
            checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
            
            # 오래된 체크포인트 삭제
            for checkpoint in checkpoints[:-self.config.save_total_limit]:
                import shutil
                shutil.rmtree(checkpoint)
                self.logger.info(f"🗑️ 체크포인트 삭제: {checkpoint}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint_path = Path(checkpoint_path)
        
        # 모델 로드
        self.model = PeftModel.from_pretrained(
            self.model, 
            checkpoint_path,
            device_map={"": self.device}
        )
        
        # 훈련 상태 로드
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_metric = state['best_metric']
            
            if self.optimizer:
                self.optimizer.load_state_dict(state['optimizer'])
            if self.scheduler:
                self.scheduler.load_state_dict(state['scheduler'])
            if self.config.fp16 and self.scaler:
                self.scaler.load_state_dict(state['scaler'])
        
        self.logger.info(f"📥 체크포인트 로드: {checkpoint_path}")
    
    def train(self):
        """메인 훈련 루프"""
        self.logger.info("🚀 훈련 시작")
        
        # 설정
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        # 체크포인트에서 재시작
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        start_time = time.time()
        
        try:
            while self.global_step < self.config.max_steps:
                epoch_loss = self.train_epoch()
                self.epoch += 1
                
                if self.is_master:
                    self.logger.info(f"Epoch {self.epoch} 완료, 평균 손실: {epoch_loss:.4f}")
                
                # 최대 스텝 도달 시 종료
                if self.global_step >= self.config.max_steps:
                    break
        
        except KeyboardInterrupt:
            self.logger.info("⚠️ 훈련이 중단되었습니다")
        
        finally:
            # 최종 체크포인트 저장
            if self.is_master:
                self.save_checkpoint()
            
            # 분산 프로세스 정리
            if self.world_size > 1:
                dist.destroy_process_group()
        
        total_time = time.time() - start_time
        self.logger.info(f"✅ 훈련 완료 (시간: {total_time:.2f}초)")
        
        if self.writer:
            self.writer.close()
    
    def evaluate_model(self):
        """모델 평가 (추론 품질 테스트)"""
        self.logger.info("📊 모델 평가 시작")
        
        # 설정
        self.setup_model()
        
        # 베스트 모델 로드
        best_model_path = self.exp_dir / "best_model"
        if best_model_path.exists():
            self.load_checkpoint(best_model_path)
        
        # 평가 실행
        results = evaluate_model(self.pipeline, self.config)
        
        # 결과 저장
        results_path = self.exp_dir / "results" / "evaluation_results.json"
        results_path.parent.mkdir(exist_ok=True)
        
        import json
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"📊 평가 완료: {results_path}")
        return results