#!/usr/bin/env python3
"""
TRELLIS 양자화 실험 - 메인 실행 파일

Usage:
    python main.py --model text-base
    python main.py --model text-large --output_dir my_results
"""

import argparse
import os
import sys
from pathlib import Path

# TRELLIS 프로젝트 루트를 Python 경로에 추가
script_dir = Path(__file__).parent
trellis_root = script_dir.parent  # 상위 디렉토리 (TRELLIS 프로젝트 루트)
sys.path.insert(0, str(script_dir))  # 현재 디렉토리
sys.path.insert(0, str(trellis_root))  # TRELLIS 루트

from quantization_manager import TRELLISQuantizationManager


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="TRELLIS 모델 양자화 실험")
    parser.add_argument("--model", type=str, default="text-base",
                        choices=["text-base", "text-large", "text-xlarge", "image-large"],
                        help="TRELLIS 모델 선택 (자동으로 경로 매핑됨)")
    parser.add_argument("--output_dir", type=str, default="quantization_results",
                        help="결과 저장 디렉토리")
    parser.add_argument("--model_path", type=str, default=None,
                        help="커스텀 모델 경로 (지정하지 않으면 --model로 자동 설정)")
    
    args = parser.parse_args()
    
    print("🔧 TRELLIS 양자화 실험")
    print("=" * 40)
    print("📋 주요 기능:")
    print("✅ Dynamic INT8 양자화")
    print("✅ 성능 비교 (파라미터/크기/메모리)")
    print("✅ 양자화 모델 저장")
    print("✅ 품질 테스트 도구")
    print("=" * 40)
    
    # 양자화 매니저 생성
    manager = TRELLISQuantizationManager(
        model_name=args.model,
        output_dir=args.output_dir
    )
    
    # 커스텀 경로 처리
    if args.model_path:
        if os.path.exists(args.model_path):
            manager.model_path = args.model_path
            print(f"🔧 커스텀 모델 경로: {args.model_path}")
        else:
            print(f"⚠️ 커스텀 경로가 존재하지 않습니다: {args.model_path}")
            print(f"🔄 기본 경로 사용: {manager.model_path}")
    else:
        print(f"📂 자동 매핑 경로: {manager.model_path}")
    
    # 실험 실행
    success = manager.run_experiment()
    
    if success:
        print("\n🎉 실험 완료!")
        print(f"📊 결과 확인: {manager.output_dir}/")
        print(f"🔧 품질 테스트: python model_test.py quick")
        exit(0)
    else:
        print("\n❌ 실험 실패")
        exit(1)


if __name__ == "__main__":
    main()