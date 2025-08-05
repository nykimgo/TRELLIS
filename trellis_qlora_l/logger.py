"""
로깅 유틸리티

주요 기능:
- 파일 및 콘솔 로깅
- 다양한 로그 레벨 지원
- 분산 훈련용 로깅
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
import datetime


def setup_logger(
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    rank: int = 0,
    world_size: int = 1
) -> logging.Logger:
    """로거 설정"""
    
    # 로거 생성
    logger = logging.getLogger("TRELLISQLoRA")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    if world_size > 1:
        formatter = logging.Formatter(
            f'[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # 콘솔 핸들러 (rank 0만 또는 단일 프로세스)
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 파일 핸들러
    if log_file and rank == 0:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 전파 방지
    logger.propagate = False
    
    return logger


class TrainingLogger:
    """훈련 전용 로거"""
    
    def __init__(self, exp_dir: Path, rank: int = 0, world_size: int = 1):
        self.exp_dir = exp_dir
        self.rank = rank
        self.world_size = world_size
        self.is_master = (rank == 0)
        
        # 로그 디렉토리 생성
        self.log_dir = exp_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 로거 설정
        self.logger = setup_logger(
            log_file=self.log_dir / "training.log" if self.is_master else None,
            rank=rank,
            world_size=world_size
        )
        
        # 메트릭 로그 파일
        if self.is_master:
            self.metrics_file = self.log_dir / "metrics.csv"
            self._init_metrics_file()
    
    def _init_metrics_file(self):
        """메트릭 CSV 파일 초기화"""
        if not self.metrics_file.exists():
            with open(self.metrics_file, 'w') as f:
                f.write("step,epoch,train_loss,eval_loss,learning_rate,timestamp\n")
    
    def info(self, message: str):
        """정보 로그"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """경고 로그"""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """에러 로그"""
        self.logger.error(message, exc_info=exc_info)
    
    def debug(self, message: str):
        """디버그 로그"""
        self.logger.debug(message)
    
    def log_metrics(self, step: int, epoch: int, metrics: dict):
        """메트릭 로깅"""
        if not self.is_master:
            return
        
        # 콘솔 출력
        metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in metrics.items()])
        self.info(f"Step {step:6d} | Epoch {epoch:3d} | {metric_str}")
        
        # CSV 파일에 기록
        timestamp = datetime.datetime.now().isoformat()
        train_loss = metrics.get('train_loss', '')
        eval_loss = metrics.get('eval_loss', '')
        lr = metrics.get('learning_rate', '')
        
        with open(self.metrics_file, 'a') as f:
            f.write(f"{step},{epoch},{train_loss},{eval_loss},{lr},{timestamp}\n")
    
    def log_model_info(self, model_info: dict):
        """모델 정보 로깅"""
        if not self.is_master:
            return
        
        self.info("📊 모델 정보:")
        for key, value in model_info.items():
            if isinstance(value, dict):
                self.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.info(f"    {sub_key}: {sub_value}")
            else:
                self.info(f"  {key}: {value}")
    
    def log_config(self, config):
        """설정 정보 로깅"""
        if not self.is_master:
            return
        
        self.info("⚙️ 훈련 설정:")
        config_dict = config.__dict__ if hasattr(config, '__dict__') else config
        for key, value in config_dict.items():
            if not key.startswith('_'):
                self.info(f"  {key}: {value}")
    
    def log_system_info(self):
        """시스템 정보 로깅"""
        if not self.is_master:
            return
        
        import torch
        import platform
        
        self.info("💻 시스템 정보:")
        self.info(f"  Python: {platform.python_version()}")
        self.info(f"  PyTorch: {torch.__version__}")
        self.info(f"  CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
        self.info(f"  GPU 수: {torch.cuda.device_count()}")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                self.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    def save_log_summary(self):
        """로그 요약 저장"""
        if not self.is_master:
            return
        
        summary_file = self.log_dir / "summary.txt"
        
        # 메트릭 파일에서 통계 계산
        if self.metrics_file.exists():
            import pandas as pd
            try:
                df = pd.read_csv(self.metrics_file)
                
                with open(summary_file, 'w') as f:
                    f.write("=== 훈련 요약 ===\n")
                    f.write(f"총 스텝: {df['step'].max() if not df.empty else 0}\n")
                    f.write(f"총 에포크: {df['epoch'].max() if not df.empty else 0}\n")
                    
                    if 'train_loss' in df.columns and not df['train_loss'].isna().all():
                        f.write(f"최종 훈련 손실: {df['train_loss'].iloc[-1]:.4f}\n")
                        f.write(f"최소 훈련 손실: {df['train_loss'].min():.4f}\n")
                    
                    if 'eval_loss' in df.columns and not df['eval_loss'].isna().all():
                        f.write(f"최종 검증 손실: {df['eval_loss'].dropna().iloc[-1]:.4f}\n")
                        f.write(f"최소 검증 손실: {df['eval_loss'].min():.4f}\n")
                    
                    f.write(f"시작 시간: {df['timestamp'].iloc[0] if not df.empty else 'N/A'}\n")
                    f.write(f"종료 시간: {df['timestamp'].iloc[-1] if not df.empty else 'N/A'}\n")
                
                self.info(f"📊 훈련 요약 저장: {summary_file}")
                
            except Exception as e:
                self.error(f"요약 저장 실패: {e}")


class ProgressLogger:
    """진행상황 로거"""
    
    def __init__(self, total_steps: int, log_interval: int = 100):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.start_time = None
        self.step_times = []
    
    def start(self):
        """시작 시간 기록"""
        import time
        self.start_time = time.time()
    
    def log_progress(self, current_step: int, loss: float = None, lr: float = None):
        """진행상황 로깅"""
        if current_step % self.log_interval != 0:
            return
        
        import time
        current_time = time.time()
        
        if self.start_time:
            elapsed = current_time - self.start_time
            steps_per_sec = current_step / elapsed if elapsed > 0 else 0
            eta = (self.total_steps - current_step) / steps_per_sec if steps_per_sec > 0 else 0
            
            progress = current_step / self.total_steps * 100
            
            message = f"Progress: {progress:5.1f}% ({current_step}/{self.total_steps})"
            message += f" | Speed: {steps_per_sec:.2f} steps/s"
            message += f" | ETA: {eta/3600:.1f}h"
            
            if loss is not None:
                message += f" | Loss: {loss:.4f}"
            
            if lr is not None:
                message += f" | LR: {lr:.2e}"
            
            print(f"\r{message}", end="", flush=True)
    
    def finish(self):
        """완료 처리"""
        print("\n✅ 훈련 완료!")


class MetricsTracker:
    """메트릭 추적 클래스"""
    
    def __init__(self):
        self.metrics = {}
        self.history = {}
    
    def update(self, **kwargs):
        """메트릭 업데이트"""
        for key, value in kwargs.items():
            self.metrics[key] = value
            
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_average(self, key: str, last_n: int = None) -> float:
        """평균값 계산"""
        if key not in self.history:
            return 0.0
        
        values = self.history[key]
        if last_n:
            values = values[-last_n:]
        
        return sum(values) / len(values) if values else 0.0
    
    def get_best(self, key: str, mode: str = "min") -> float:
        """최적값 반환"""
        if key not in self.history:
            return float('inf') if mode == "min" else float('-inf')
        
        values = self.history[key]
        return min(values) if mode == "min" else max(values)
    
    def get_current(self, key: str) -> float:
        """현재값 반환"""
        return self.metrics.get(key, 0.0)
    
    def get_trend(self, key: str, window: int = 10) -> str:
        """트렌드 분석"""
        if key not in self.history or len(self.history[key]) < window:
            return "stable"
        
        recent = self.history[key][-window:]
        older = self.history[key][-window*2:-window] if len(self.history[key]) >= window*2 else recent[:window//2]
        
        if not older:
            return "stable"
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        diff_ratio = abs(recent_avg - older_avg) / older_avg if older_avg != 0 else 0
        
        if diff_ratio < 0.01:
            return "stable"
        elif recent_avg < older_avg:
            return "improving"
        else:
            return "degrading"
    
    def save_to_file(self, file_path: Path):
        """메트릭을 파일로 저장"""
        import json
        
        with open(file_path, 'w') as f:
            json.dump({
                'current': self.metrics,
                'history': self.history
            }, f, indent=2)
    
    def load_from_file(self, file_path: Path):
        """파일에서 메트릭 로드"""
        import json
        
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.metrics = data.get('current', {})
                self.history = data.get('history', {})


def create_experiment_logger(exp_dir: Path, config, rank: int = 0, world_size: int = 1) -> TrainingLogger:
    """실험용 로거 생성"""
    logger = TrainingLogger(exp_dir, rank, world_size)
    
    if rank == 0:
        logger.info("🚀 TRELLIS QLoRA 실험 시작")
        logger.log_system_info()
        logger.log_config(config)
    
    return logger