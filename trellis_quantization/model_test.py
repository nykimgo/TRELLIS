#!/usr/bin/env python3
"""
TRELLIS 모델 테스트 도구

기능:
1. quick_test() - 양자화 모델 로드 테스트
2. compare_models() - 원본 vs 양자화 모델 비교
3. batch_test() - 여러 프롬프트 배치 테스트

사용법:
    python model_test.py quick
    python model_test.py compare --prompt "a red car"
    python model_test.py batch
"""

import os
import argparse
import time
from pathlib import Path
import torch

# 환경 설정
os.environ['SPCONV_ALGO'] = 'native'

# ⚠️ 경로 설정 - 필요시 수정하세요
ORIGINAL_MODEL_PATH = "/home/sr/TRELLIS/microsoft/TRELLIS-text-base"
QUANTIZED_MODEL_PATH = "./quantization_results/trellis_text-base_quantized"

# 테스트 프롬프트들
TEST_PROMPTS = [
    "a red sports car",
    "a wooden chair", 
    "a blue coffee mug",
    "a small house",
    "a cute cat"
]

QUICK_TEST_PROMPTS = [
    "a red cube",
    "a blue sphere", 
    "a green pyramid"
]


def quick_test():
    """양자화된 모델 빠른 로드 테스트"""
    try:
        print("📥 양자화된 모델 로드 테스트...")
        print(f"📂 모델 경로: {QUANTIZED_MODEL_PATH}")
        
        from trellis.pipelines import TrellisTextTo3DPipeline
        
        pipeline = TrellisTextTo3DPipeline.from_pretrained(QUANTIZED_MODEL_PATH)
        pipeline.cuda()
        
        print("✅ 모델 로드 성공!")
        
        # 모델 정보 출력
        if hasattr(pipeline, 'models'):
            print("📊 모델 컴포넌트:")
            for name, model in pipeline.models.items():
                if model is not None:
                    param_count = sum(p.numel() for p in model.parameters())
                    
                    # 양자화 상태 확인
                    is_quantized = any(
                        hasattr(m, '_packed_params') or 'quantized' in str(type(m)).lower()
                        for m in model.modules()
                    )
                    status = "🔧INT8" if is_quantized else "📏FP32"
                    
                    print(f"  - {name}: {param_count/1e6:.1f}M {status}")
        
        print("\n✅ 테스트 완료!")
        print("💡 실제 3D 생성은 'python model_test.py compare' 사용")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        print("\n🔧 해결 방법:")
        print("1. TRELLIS 프로젝트 루트에서 실행하고 있는지 확인")
        print("2. QUANTIZED_MODEL_PATH 경로가 올바른지 확인")
        print("3. 양자화 실험이 성공적으로 완료되었는지 확인")


def compare_models(prompt: str, seed: int = 42):
    """원본과 양자화 모델 비교"""
    print(f"🎯 테스트: '{prompt}' (seed: {seed})")
    print(f"📂 원본 경로: {ORIGINAL_MODEL_PATH}")
    print(f"📂 양자화 경로: {QUANTIZED_MODEL_PATH}")
    
    try:
        from trellis.pipelines import TrellisTextTo3DPipeline
        from trellis.utils import postprocessing_utils
        
        # 안전한 파일명 생성
        safe_name = prompt.replace(' ', '_').replace('/', '_')
        
        # 1. 원본 모델 테스트
        print("\n📥 원본 모델 로드 중...")
        original_pipeline = TrellisTextTo3DPipeline.from_pretrained(ORIGINAL_MODEL_PATH)
        original_pipeline.cuda()
        
        print("🔄 원본 모델 생성 중...")
        original_outputs = original_pipeline.run(prompt, seed=seed)
        
        # 원본 결과 저장
        original_glb = postprocessing_utils.to_glb(
            original_outputs['gaussian'][0], 
            original_outputs['mesh'][0],
            simplify=0.95, 
            texture_size=1024
        )
        original_file = f"original_{safe_name}_seed{seed}.glb"
        original_glb.export(original_file)
        print(f"  ✅ 원본 저장: {original_file}")
        
        # 메모리 정리
        del original_pipeline
        torch.cuda.empty_cache()
        
        # 2. 양자화 모델 테스트
        print("\n📥 양자화 모델 로드 중...")
        quantized_pipeline = TrellisTextTo3DPipeline.from_pretrained(QUANTIZED_MODEL_PATH)
        quantized_pipeline.cuda()
        
        print("🔄 양자화 모델 생성 중...")
        quantized_outputs = quantized_pipeline.run(prompt, seed=seed)
        
        # 양자화 결과 저장
        quantized_glb = postprocessing_utils.to_glb(
            quantized_outputs['gaussian'][0],
            quantized_outputs['mesh'][0],
            simplify=0.95,
            texture_size=1024
        )
        quantized_file = f"quantized_{safe_name}_seed{seed}.glb"
        quantized_glb.export(quantized_file)
        print(f"  ✅ 양자화 저장: {quantized_file}")
        
        # 메모리 정리
        del quantized_pipeline
        torch.cuda.empty_cache()
        
        # 파일 크기 비교
        original_size = Path(original_file).stat().st_size / (1024 * 1024)  # MB
        quantized_size = Path(quantized_file).stat().st_size / (1024 * 1024)  # MB
        
        print(f"\n🎉 비교 완료!")
        print(f"  📦 원본: {original_file} ({original_size:.1f}MB)")
        print(f"  📦 양자화: {quantized_file} ({quantized_size:.1f}MB)")
        print(f"  📊 파일 크기 차이: {((original_size - quantized_size) / original_size * 100):+.1f}%")
        
    except Exception as e:
        print(f"❌ 비교 실패: {e}")
        print(f"\n🔧 해결 방법:")
        print(f"1. 경로 확인:")
        print(f"   - 원본: {ORIGINAL_MODEL_PATH}")
        print(f"   - 양자화: {QUANTIZED_MODEL_PATH}")
        print(f"2. GPU 메모리 부족시: export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        print(f"3. 더 간단한 프롬프트로 테스트: 'a cube'")


def batch_test():
    """빠른 배치 테스트"""
    print("🚀 TRELLIS 배치 테스트 시작")
    print(f"📝 테스트 프롬프트: {len(QUICK_TEST_PROMPTS)}개")
    print("=" * 50)
    
    try:
        from trellis.pipelines import TrellisTextTo3DPipeline
        from trellis.utils import postprocessing_utils
        
        results = []
        
        for i, prompt in enumerate(QUICK_TEST_PROMPTS, 1):
            print(f"\n🔄 테스트 {i}/{len(QUICK_TEST_PROMPTS)}: '{prompt}'")
            
            result = {
                'prompt': prompt,
                'original_success': False,
                'quantized_success': False,
                'original_time': 0,
                'quantized_time': 0,
                'original_size': 0,
                'quantized_size': 0
            }
            
            # 원본 모델 테스트
            try:
                print("  📥 원본 모델...")
                pipeline = TrellisTextTo3DPipeline.from_pretrained(ORIGINAL_MODEL_PATH)
                pipeline.cuda()
                
                start_time = time.time()
                outputs = pipeline.run(prompt, seed=42)
                result['original_time'] = time.time() - start_time
                
                # GLB 저장
                glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0])
                file_path = f"batch_original_{prompt.replace(' ', '_')}.glb"
                glb.export(file_path)
                result['original_size'] = Path(file_path).stat().st_size / (1024 * 1024)
                result['original_success'] = True
                
                del pipeline
                torch.cuda.empty_cache()
                print("    ✅ 성공")
                
            except Exception as e:
                print(f"    ❌ 실패: {e}")
            
            # 양자화 모델 테스트  
            try:
                print("  📥 양자화 모델...")
                pipeline = TrellisTextTo3DPipeline.from_pretrained(QUANTIZED_MODEL_PATH)
                pipeline.cuda()
                
                start_time = time.time()
                outputs = pipeline.run(prompt, seed=42)
                result['quantized_time'] = time.time() - start_time
                
                # GLB 저장
                glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0])
                file_path = f"batch_quantized_{prompt.replace(' ', '_')}.glb"
                glb.export(file_path)
                result['quantized_size'] = Path(file_path).stat().st_size / (1024 * 1024)
                result['quantized_success'] = True
                
                del pipeline
                torch.cuda.empty_cache()
                print("    ✅ 성공")
                
            except Exception as e:
                print(f"    ❌ 실패: {e}")
            
            results.append(result)
        
        # 결과 요약
        print(f"\n📊 배치 테스트 결과 요약")
        print("=" * 50)
        
        success_count = sum(1 for r in results if r['original_success'] and r['quantized_success'])
        print(f"✅ 성공: {success_count}/{len(results)}")
        
        if success_count > 0:
            avg_time_original = sum(r['original_time'] for r in results if r['original_success']) / success_count
            avg_time_quantized = sum(r['quantized_time'] for r in results if r['quantized_success']) / success_count
            avg_size_original = sum(r['original_size'] for r in results if r['original_success']) / success_count
            avg_size_quantized = sum(r['quantized_size'] for r in results if r['quantized_success']) / success_count
            
            time_change = ((avg_time_quantized - avg_time_original) / avg_time_original) * 100
            size_change = ((avg_size_quantized - avg_size_original) / avg_size_original) * 100
            
            print(f"⏱️ 평균 생성시간: 원본 {avg_time_original:.1f}s → 양자화 {avg_time_quantized:.1f}s ({time_change:+.1f}%)")
            print(f"📦 평균 파일크기: 원본 {avg_size_original:.1f}MB → 양자화 {avg_size_quantized:.1f}MB ({size_change:+.1f}%)")
        
        print(f"\n📁 생성된 파일들:")
        for result in results:
            if result['original_success']:
                print(f"  - batch_original_{result['prompt'].replace(' ', '_')}.glb")
            if result['quantized_success']:
                print(f"  - batch_quantized_{result['prompt'].replace(' ', '_')}.glb")
        
    except Exception as e:
        print(f"❌ 배치 테스트 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="TRELLIS 모델 테스트 도구")
    subparsers = parser.add_subparsers(dest='command', help='테스트 명령')
    
    # quick 명령
    quick_parser = subparsers.add_parser('quick', help='빠른 로드 테스트')
    
    # compare 명령
    compare_parser = subparsers.add_parser('compare', help='모델 비교')
    compare_parser.add_argument('--prompt', type=str, help='테스트할 프롬프트')
    compare_parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    compare_parser.add_argument('--all_prompts', action='store_true', help='모든 테스트 프롬프트 실행')
    
    # batch 명령
    batch_parser = subparsers.add_parser('batch', help='배치 테스트')
    
    args = parser.parse_args()
    
    if args.command == 'quick':
        quick_test()
        
    elif args.command == 'compare':
        if args.all_prompts:
            for i, prompt in enumerate(TEST_PROMPTS, 1):
                print(f"\n{'='*50}")
                print(f"🔄 테스트 {i}/{len(TEST_PROMPTS)}")
                print(f"{'='*50}")
                compare_models(prompt, args.seed)
        elif args.prompt:
            compare_models(args.prompt, args.seed)
        else:
            print("사용법:")
            print("  python model_test.py compare --prompt 'a red car'")
            print("  python model_test.py compare --all_prompts")
            print(f"\n📋 사용 가능한 테스트 프롬프트:")
            for prompt in TEST_PROMPTS:
                print(f"  - \"{prompt}\"")
                
    elif args.command == 'batch':
        batch_test()
        
    else:
        print("TRELLIS 모델 테스트 도구")
        print("=" * 30)
        print("사용법:")
        print("  python model_test.py quick                    # 빠른 로드 테스트")
        print("  python model_test.py compare --prompt 'a car' # 모델 비교")
        print("  python model_test.py batch                    # 배치 테스트")
        print()
        print("⚠️ 경로 설정:")
        print(f"  원본 모델: {ORIGINAL_MODEL_PATH}")
        print(f"  양자화 모델: {QUANTIZED_MODEL_PATH}")
        print("  필요시 스크립트 상단에서 수정하세요")


if __name__ == "__main__":
    main()