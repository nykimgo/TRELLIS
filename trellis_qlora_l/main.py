#!/usr/bin/env python3
"""
TRELLIS QLoRA 실험 - 메인 실행 파일

Usage:
    python main.py --config configs/qlora_config.yaml
    python main.py --model text-large --rank 16 --alpha 32
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# TRELLIS 프로젝트 루트를 Python 경로에 추가
script_dir = Path(__file__).parent
trellis_root = script_dir.parent  # 상위 디렉토리 (TRELLIS 프로젝트 루트)
sys.path.insert(0, str(script_dir))  # 현재 디렉토리
sys.path.insert(0, str(trellis_root))  # TRELLIS 루트

from qlora_trainer import TRELLISQLoRATrainer
from qlora_config import QLoRAConfig
from utils.logger import setup_logger


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="TRELLIS QLoRA Fine-tuning")
    
    # 기본 설정
    parser.add_argument("--config", type=str, default="configs/qlora_config.yaml",
                        help="QLoRA 설정 파일 경로")
    parser.add_argument("--model", type=str, default="text-large",
                        choices=["text-base", "text-large", "text-xlarge"],
                        help="TRELLIS 모델 선택")
    parser.add_argument("--output_dir", type=str, default="./qlora_experiments",
                        help="실험 결과 저장 디렉토리")
    
    # QLoRA 설정
    parser.add_argument("--rank", type=int, default=16,
                        help="LoRA rank (r)")
    parser.add_argument("--alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="LoRA dropout")
    parser.add_argument("--target_modules", nargs="+", 
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="LoRA 적용 대상 모듈")
    
    # 훈련 설정
    parser.add_argument("--batch_size", type=int, default=2,
                        help="배치 크기")
    parser.add_argument("--batch_size_per_gpu", type=int, default=None,
                        help="GPU당 배치 크기")
    parser.add_argument("--max_steps", type=int, default=10000,
                        help="최대 훈련 스텝")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="학습률")
    parser.add_argument("--save_steps", type=int, default=1000,
                        help="체크포인트 저장 간격")
    
    # 시스템 설정
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="사용할 GPU 수")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="Mixed Precision 사용")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Gradient Checkpointing 사용")
    
    # 데이터셋 설정
    parser.add_argument("--dataset_path", type=str, default="",
                        help="데이터셋 경로 (HSSD 루트 디렉토리)")
    parser.add_argument("--dataset_type", type=str, default="auto",
                        choices=["auto", "hssd", "custom", "dummy"],
                        help="데이터셋 타입")
    parser.add_argument("--hssd_splits", nargs="+", default=["train", "val"],
                        help="HSSD 사용할 데이터 분할")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="체크포인트에서 재시작")
    parser.add_argument("--eval_only", action="store_true", default=False,
                        help="평가만 실행")
    
    args = parser.parse_args()
    return args


def load_config(config_path: str, args) -> QLoRAConfig:
    """설정 파일 로드 및 명령행 인자로 덮어쓰기"""
    config = QLoRAConfig()
    
    # YAML 설정 파일 로드
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        config.update_from_dict(yaml_config)
        print(f"📄 설정 파일 로드: {config_path}")
    else:
        print(f"⚠️ 설정 파일 없음: {config_path}, 기본값 사용")
    
    # 명령행 인자로 덮어쓰기
    config.update_from_args(args)
    
    return config


def setup_experiment_dir(config: QLoRAConfig) -> Path:
    """실험 디렉토리 설정"""
    if config.experiment_name is None:
        # 자동 실험 이름 생성
        config.experiment_name = f"qlora_{config.model_name}_r{config.lora_rank}_a{config.lora_alpha}"
    
    exp_dir = Path(config.output_dir) / config.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 하위 디렉토리 생성
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "tensorboard").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    
    # 설정 파일 저장
    config_file = exp_dir / "config.yaml"
    config.save_to_file(config_file)
    
    print(f"🔧 실험 디렉토리: {exp_dir}")
    return exp_dir


def main():
    """메인 실행 함수"""
    print("🚀 TRELLIS QLoRA Fine-tuning")
    print("=" * 50)
    
    # 인자 파싱
    args = parse_args()
    
    # 설정 로드
    config = load_config(args.config, args)
    
    # 실험 디렉토리 설정
    exp_dir = setup_experiment_dir(config)
    
    # 로거 설정
    logger = setup_logger(exp_dir / "logs" / "training.log")
    logger.info("🔧 TRELLIS QLoRA 실험 시작")
    logger.info(f"📋 설정: {config}")
    
    # 환경 변수 설정
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    if config.num_gpus > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(config.num_gpus))
    
    try:
        # QLoRA 트레이너 생성
        trainer = TRELLISQLoRATrainer(config, exp_dir, logger)
        
        if args.eval_only:
            print("📊 평가 모드")
            trainer.evaluate()
        else:
            print("🎯 훈련 시작")
            trainer.train()
            
        print("\n✅ 실험 완료!")
        print(f"📊 결과 확인: {exp_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 중단됨")
        logger.info("훈련이 사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 실험 실패: {e}")
        logger.error(f"실험 실패: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()