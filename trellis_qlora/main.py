#!/usr/bin/env python3
"""TRELLIS QLoRA 실험 메인 실행 파일"""

import argparse
from pathlib import Path
import sys

# 현재 디렉토리와 상위 TRELLIS 루트 경로를 파이썬 경로에 추가
SCRIPT_DIR = Path(__file__).parent
TRELLIS_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TRELLIS_ROOT))

from qlora_manager import TRELLISQLoRAManager


def parse_args() -> argparse.Namespace:
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="TRELLIS 모델 QLoRA 미세튜닝")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/TRELLIS/microsoft/TRELLIS-text-large/ckpts",
        help="사전학습된 TRELLIS 모델 경로",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/home/sr/TRELLIS/datasets/HSSD",
        help="학습 데이터셋 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qlora_results",
        help="체크포인트 및 로그 저장 디렉토리",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="총 학습 스텝 수",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="GPU 당 배치 크기",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="학습률",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="체크포인트 저장 주기",
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help="텐서보드 로그 주기",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("🔧 TRELLIS QLoRA 미세튜닝")
    print("=" * 40)
    print(f"📂 모델 경로: {args.model_path}")
    print(f"📁 데이터셋: {args.dataset_path}")
    print(f"📁 출력 디렉토리: {args.output_dir}")
    print("=" * 40)

    manager = TRELLISQLoRAManager(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        save_steps=args.save_steps,
        log_steps=args.log_steps,
    )

    success = manager.run_experiment()

    if success:
        print("\n🎉 QLoRA 학습 완료")
        print(f"📦 체크포인트: {args.output_dir}/ckpts")
        print(f"📈 텐서보드 로그: {args.output_dir}/logs")
        sys.exit(0)
    else:
        print("\n❌ QLoRA 학습 실패")
        sys.exit(1)


if __name__ == "__main__":
    main()
