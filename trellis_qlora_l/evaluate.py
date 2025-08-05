#!/usr/bin/env python3
"""
TRELLIS QLoRA 모델 평가 스크립트

Usage:
    python scripts/evaluate.py --model_path ./qlora_experiments/qlora_text-large_r16_a32/best_model
    python scripts/evaluate.py --compare --original_path /path/to/original --qlora_path /path/to/qlora
"""

import argparse
import os
import sys
from pathlib import Path
import json

# 경로 추가
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir.parent))

from utils.evaluation import run_comprehensive_evaluation, compare_models
from qlora_config import QLoRAConfig

try:
    from trellis.pipelines import TrellisTextTo3DPipeline
    from peft import PeftModel
except ImportError as e:
    print(f"❌ 임포트 오류: {e}")
    print("TRELLIS와 PEFT 라이브러리가 필요합니다.")
    sys.exit(1)


def load_qlora_model(model_path: str, base_model_path: str = None):
    """QLoRA 모델 로드"""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"모델 경로가 존재하지 않습니다: {model_path}")
    
    print(f"📥 QLoRA 모델 로드: {model_path}")
    
    # 설정 파일에서 원본 모델 경로 찾기
    if base_model_path is None:
        config_file = model_path.parent / "config.yaml"
        if config_file.exists():
            config = QLoRAConfig()
            import yaml
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            config.update_from_dict(config_dict)
            base_model_path = config.model_path
        else:
            # 기본 경로 추정
            base_model_path = "/home/sr/TRELLIS/microsoft/TRELLIS-text-large"
    
    print(f"📂 베이스 모델: {base_model_path}")
    
    # 베이스 파이프라인 로드
    pipeline = TrellisTextTo3DPipeline.from_pretrained(base_model_path)
    
    # QLoRA 어댑터 로드
    if hasattr(pipeline, 'sparse_structure_decoder'):
        base_model = pipeline.sparse_structure_decoder
    else:
        base_model = next(iter(pipeline.models.values()))
    
    # LoRA 어댑터 적용
    qlora_model = PeftModel.from_pretrained(base_model, model_path)
    
    # 파이프라인에 적용
    if hasattr(pipeline, 'sparse_structure_decoder'):
        pipeline.sparse_structure_decoder = qlora_model
    
    return pipeline, qlora_model


def evaluate_single_model(args):
    """단일 모델 평가"""
    print("🔍 단일 모델 평가")
    
    # 모델 로드
    pipeline, model = load_qlora_model(args.model_path, args.base_model_path)
    
    # 설정 로드
    config_path = Path(args.model_path).parent / "config.yaml"
    if config_path.exists():
        config = QLoRAConfig()
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config.update_from_dict(config_dict)
    else:
        # 기본 설정
        config = QLoRAConfig()
    
    # 평가 실행
    results_dir = Path(args.output_dir) / "evaluation_results"
    results = run_comprehensive_evaluation(pipeline, config, results_dir)
    
    print("\n📊 평가 결과:")
    if 'quality_metrics' in results:
        for metric, value in results['quality_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\n📄 상세 결과: {results_dir}")


def compare_two_models(args):
    """두 모델 비교"""
    print("🔄 모델 비교 평가")
    
    # 원본 모델 로드
    print("📥 원본 모델 로드...")
    original_pipeline = TrellisTextTo3DPipeline.from_pretrained(args.original_path)
    
    # QLoRA 모델 로드
    print("📥 QLoRA 모델 로드...")
    qlora_pipeline, _ = load_qlora_model(args.qlora_path, args.original_path)
    
    # 설정 (QLoRA 모델 기준)
    config_path = Path(args.qlora_path).parent / "config.yaml"
    if config_path.exists():
        config = QLoRAConfig()
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        config.update_from_dict(config_dict)
    else:
        config = QLoRAConfig()
    
    # 각각 평가
    print("🔍 원본 모델 평가...")
    original_results = run_comprehensive_evaluation(
        original_pipeline, config, 
        Path(args.output_dir) / "original_evaluation"
    )
    
    print("🔍 QLoRA 모델 평가...")
    qlora_results = run_comprehensive_evaluation(
        qlora_pipeline, config,
        Path(args.output_dir) / "qlora_evaluation"
    )
    
    # 비교 분석
    print("📊 비교 분석...")
    comparison = compare_models(
        original_results['quality_metrics'],
        qlora_results['quality_metrics']
    )
    
    # 비교 결과 저장
    comparison_file = Path(args.output_dir) / "model_comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump({
            'original_results': original_results,
            'qlora_results': qlora_results,
            'comparison': comparison
        }, f, indent=2)
    
    # 결과 출력
    print("\n📊 비교 결과:")
    for metric, data in comparison['metrics_comparison'].items():
        print(f"  {metric}:")
        print(f"    원본: {data['original']:.4f}")
        print(f"    QLoRA: {data['qlora']:.4f}")
        print(f"    개선: {data['improvement_percent']:+.2f}%")
    
    print(f"\n📄 상세 비교: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description="TRELLIS QLoRA 모델 평가")
    
    # 공통 인자
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="평가 결과 저장 디렉토리")
    
    # 단일 모델 평가
    parser.add_argument("--model_path", type=str, default=None,
                        help="평가할 QLoRA 모델 경로")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="베이스 모델 경로 (자동 탐지하지 못할 경우)")
    
    # 모델 비교
    parser.add_argument("--compare", action="store_true",
                        help="두 모델 비교 모드")
    parser.add_argument("--original_path", type=str, default=None,
                        help="원본 모델 경로")
    parser.add_argument("--qlora_path", type=str, default=None,
                        help="QLoRA 모델 경로")
    
    # 실행 옵션
    parser.add_argument("--quick", action="store_true",
                        help="빠른 평가 (적은 샘플 수)")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # GPU 설정
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 평가는 단일 GPU 사용
    
    try:
        if args.compare:
            # 비교 모드
            if not args.original_path or not args.qlora_path:
                print("❌ 비교 모드에서는 --original_path와 --qlora_path가 필요합니다.")
                sys.exit(1)
            compare_two_models(args)
        
        else:
            # 단일 모델 평가
            if not args.model_path:
                print("❌ --model_path가 필요합니다.")
                sys.exit(1)
            evaluate_single_model(args)
    
    except Exception as e:
        print(f"❌ 평가 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()