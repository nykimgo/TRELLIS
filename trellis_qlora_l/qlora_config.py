"""
TRELLIS QLoRA 설정 클래스

주요 기능:
- QLoRA 및 훈련 파라미터 관리
- YAML 설정 파일 지원
- 명령행 인자 통합
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class QLoRAConfig:
    """QLoRA 및 훈련 설정"""
    
    # 기본 모델 설정
    model_name: str = "text-large"
    model_path: str = ""
    output_dir: str = "./qlora_experiments"
    experiment_name: Optional[str] = None
    
    # QLoRA 설정
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    lora_bias: str = "none"  # none, all, lora_only
    
    # 양자화 설정
    quantization_config: Dict[str, Any] = field(default_factory=lambda: {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4"
    })
    
    # 훈련 설정
    batch_size: int = 2
    batch_size_per_gpu: Optional[int] = None
    gradient_accumulation_steps: int = 4
    max_steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    
    # 옵티마이저 설정
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # 스케줄러 설정
    lr_scheduler_type: str = "cosine"
    cosine_annealing_min_lr: float = 1e-6
    
    # 정규화 설정
    gradient_clipping: float = 1.0
    
    # 시스템 설정
    num_gpus: int = 4
    fp16: bool = True
    fp16_opt_level: str = "O1"
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # 체크포인트 설정
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None
    
    # 로깅 설정
    logging_steps: int = 100
    eval_steps: int = 1000
    use_tensorboard: bool = True
    log_level: str = "INFO"
    
    # 데이터셋 설정
    dataset_path: str = ""
    dataset_type: str = "auto"  # auto, hssd, custom, dummy
    max_seq_length: int = 2048
    dataset_split_ratio: float = 0.9  # train/val split
    
    # HSSD 전용 설정
    hssd_config: Dict[str, Any] = field(default_factory=lambda: {
        "use_rendered": True,     # 렌더링된 이미지 사용
        "use_voxelized": True,    # 복셀 데이터 사용  
        "use_latent": True,       # 인코딩된 latent 사용
        "splits": ["train", "val"], # 사용할 데이터 분할
        "max_samples_per_split": None,  # 분할당 최대 샘플 수 (None=모두)
        "cache_data": False       # 데이터 캐싱 여부
    })
    
    # EMA 설정
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # 기타 설정
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """초기화 후 처리"""
        self._setup_model_paths()
        self._validate_config()
    
    def _setup_model_paths(self):
        """모델 경로 자동 설정"""
        if not self.model_path:
            # 기본 경로 매핑
            model_paths = {
                "text-base": "/home/sr/TRELLIS/microsoft/TRELLIS-text-base",
                "text-large": "/home/sr/TRELLIS/microsoft/TRELLIS-text-large", 
                "text-xlarge": "/home/sr/TRELLIS/microsoft/TRELLIS-text-xlarge"
            }
            
            if self.model_name in model_paths:
                self.model_path = model_paths[self.model_name]
            else:
                self.model_path = self.model_name
    
    def _validate_config(self):
        """설정 검증"""
        # 경로 존재 확인
        if self.model_path and not os.path.exists(self.model_path):
            print(f"⚠️ 모델 경로가 존재하지 않습니다: {self.model_path}")
        
        # LoRA 파라미터 검증
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank는 양수여야 합니다: {self.lora_rank}")
        
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha는 양수여야 합니다: {self.lora_alpha}")
        
        # GPU 설정 검증
        if self.num_gpus <= 0:
            raise ValueError(f"num_gpus는 양수여야 합니다: {self.num_gpus}")
        
        # 배치 크기 자동 조정
        if self.batch_size_per_gpu is None:
            self.batch_size_per_gpu = max(1, self.batch_size // self.num_gpus)
        
        # gradient_accumulation_steps 자동 조정
        if self.batch_size > self.batch_size_per_gpu * self.num_gpus:
            self.gradient_accumulation_steps = max(1, 
                self.batch_size // (self.batch_size_per_gpu * self.num_gpus))
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """딕셔너리에서 설정 업데이트"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 재검증
        self._setup_model_paths()
        self._validate_config()
    
    def update_from_args(self, args):
        """명령행 인자에서 설정 업데이트"""
        arg_dict = vars(args)
        for key, value in arg_dict.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        
        # 재검증
        self._setup_model_paths()
        self._validate_config()
    
    def save_to_file(self, file_path: Path):
        """설정을 YAML 파일로 저장"""
        config_dict = asdict(self)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def get_effective_batch_size(self) -> int:
        """실제 배치 크기 계산"""
        return self.batch_size_per_gpu * self.num_gpus * self.gradient_accumulation_steps
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "effective_batch_size": self.get_effective_batch_size()
        }
    
    def __str__(self) -> str:
        """설정 요약 출력"""
        lines = []
        lines.append(f"QLoRA Configuration:")
        lines.append(f"  Model: {self.model_name} ({self.model_path})")
        lines.append(f"  LoRA: r={self.lora_rank}, α={self.lora_alpha}, dropout={self.lora_dropout}")
        lines.append(f"  Target modules: {self.lora_target_modules}")
        lines.append(f"  Batch size: {self.batch_size} (per_gpu: {self.batch_size_per_gpu}, effective: {self.get_effective_batch_size()})")
        lines.append(f"  Learning rate: {self.learning_rate}")
        lines.append(f"  Max steps: {self.max_steps}")
        lines.append(f"  GPUs: {self.num_gpus}, FP16: {self.fp16}")
        lines.append(f"  Gradient checkpointing: {self.gradient_checkpointing}")
        return "\n".join(lines)


def create_default_config() -> QLoRAConfig:
    """기본 설정 생성"""
    return QLoRAConfig()


def load_config_from_file(file_path: str) -> QLoRAConfig:
    """파일에서 설정 로드"""
    config = QLoRAConfig()
    
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        config.update_from_dict(config_dict)
    
    return config