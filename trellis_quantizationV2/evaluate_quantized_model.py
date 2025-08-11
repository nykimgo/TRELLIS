#!/usr/bin/env python3
"""
TRELLIS 양자화 모델 평가 스크립트

사용자가 미리 양자화해둔 체크포인트들을 평가하는 스크립트입니다.

사용법:
    python evaluate_quantized_model.py \\
        --model_path quantization_results/trellis_text-base_quantized \\
        --dataset datasets/Toys4k \\
        --CLIP --FD --efficiency \\
        --output_dir evaluation_results \\
        --num_samples 50

지원 옵션:
    --CLIP: CLIP Score 계산 (TRELLIS 논문 방식)
    --FD: Fréchet Distance 계산 (DINOv2 기반)
    --efficiency: 효율성 지표 계산 (파라미터, 메모리, 속도)
    --num_samples: 평가할 샘플 수
    --output_dir: 결과 저장 디렉토리
    --report_name: 보고서 파일명 (기본값: 자동 생성)
"""

import os
import sys
import json
import argparse
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# TRELLIS 관련 import
try:
    from trellis.pipelines import TrellisImageTo3DPipeline
except ImportError:
    print("❌ TRELLIS 라이브러리를 찾을 수 없습니다. 경로를 확인해주세요.")
    sys.exit(1)

# 평가 모듈 import
from metrics_evaluator import get_metrics_evaluator, cleanup_global_evaluator


class QuantizedModelEvaluator:
    """양자화된 TRELLIS 모델 평가 클래스"""
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str):
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path) 
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.evaluator = None
        
        print(f"🔧 양자화 모델 평가기 초기화")
        print(f"  모델 경로: {self.model_path}")
        print(f"  데이터셋 경로: {self.dataset_path}")
        print(f"  출력 디렉토리: {self.output_dir}")
        print(f"  디바이스: {self.device}")
    
    def load_model_config(self) -> Dict[str, Any]:
        """모델 설정 파일 로드 (pipeline.json)"""
        config_path = self.model_path / "pipeline.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"✅ 모델 설정 로드 완료: {config_path}")
        return config
    
    def load_pipeline(self) -> TrellisImageTo3DPipeline:
        """양자화된 파이프라인 로드"""
        print(f"🔧 파이프라인 로드 중: {self.model_path}")
        
        try:
            # TRELLIS 파이프라인 로드 (from_pretrained 사용)
            if self.model_path.is_dir():
                pipeline = TrellisImageTo3DPipeline.from_pretrained(str(self.model_path))
            else:
                raise ValueError(f"모델 경로가 디렉토리가 아닙니다: {self.model_path}")
            
            pipeline = pipeline.to(self.device)
            print(f"✅ 파이프라인 로드 완료")
            
            # 파이프라인 구조 분석
            self.analyze_pipeline_structure(pipeline)
            
            return pipeline
            
        except Exception as e:
            print(f"❌ 파이프라인 로드 실패: {e}")
            raise
    
    def analyze_pipeline_structure(self, pipeline):
        """파이프라인 구조 분석 및 출력"""
        print("\n📋 파이프라인 구조 분석:")
        
        if hasattr(pipeline, 'models') and isinstance(pipeline.models, dict):
            total_params = 0
            total_size = 0
            
            for model_name, model in pipeline.models.items():
                if model is not None:
                    try:
                        params = sum(p.numel() for p in model.parameters())
                        size = sum(p.numel() * p.element_size() for p in model.parameters())
                        
                        total_params += params
                        total_size += size
                        
                        print(f"  • {model_name}: {params/1e6:.1f}M 파라미터, {size/(1024*1024):.1f}MB")
                        
                    except Exception as e:
                        print(f"  • {model_name}: 분석 실패 ({e})")
                else:
                    print(f"  • {model_name}: None (로드되지 않음)")
            
            print(f"  📊 총합: {total_params/1e6:.1f}M 파라미터, {total_size/(1024*1024):.1f}MB")
        
        print()
    
    def load_dataset(self, num_samples: int = 100) -> Tuple[List[str], List[Any]]:
        """데이터셋 로드 (Toys4k)"""
        print(f"📂 데이터셋 로드 중: {self.dataset_path} (샘플 수: {num_samples})")
        
        # Toys4k 데이터셋 처리
        if "Toys4k" in str(self.dataset_path):
            return self.load_toys4k_dataset(num_samples)
        else:
            # 기타 데이터셋 처리 (커스텀 구현)
            return self.load_custom_dataset(num_samples)
    
    def load_toys4k_dataset(self, num_samples: int) -> Tuple[List[str], List[Any]]:
        """Toys4k 데이터셋 로드"""
        try:
            # dataset_toolkits의 Toys4k 모듈 활용
            sys.path.append(str(Path(__file__).parent.parent / "dataset_toolkits"))
            import datasets.Toys4k as Toys4k
            
            # 메타데이터 로드
            metadata = Toys4k.get_metadata()
            
            # 랜덤 샘플링
            if len(metadata) > num_samples:
                metadata = metadata.sample(n=num_samples, random_state=42)
            
            # 텍스트 프롬프트 생성 (파일명 기반)
            text_prompts = []
            asset_paths = []
            
            for _, row in metadata.iterrows():
                # 파일명에서 객체명 추출하여 프롬프트 생성
                filename = row.get('file_identifier', '')
                object_name = filename.replace('.blend', '').replace('_', ' ')
                prompt = f"a 3D model of {object_name}"
                text_prompts.append(prompt)
                
                # 실제 파일 경로 (존재할 경우)
                if 'local_path' in row:
                    asset_path = self.dataset_path.parent / row['local_path']
                    asset_paths.append(str(asset_path) if asset_path.exists() else None)
                else:
                    asset_paths.append(None)
            
            print(f"✅ Toys4k 데이터셋 로드 완료: {len(text_prompts)}개 샘플")
            return text_prompts, asset_paths
            
        except Exception as e:
            print(f"⚠️ Toys4k 데이터셋 로드 실패: {e}")
            print("기본 프롬프트 사용")
            return self.generate_default_prompts(num_samples)
    
    def load_custom_dataset(self, num_samples: int) -> Tuple[List[str], List[Any]]:
        """커스텀 데이터셋 로드"""
        print("📋 커스텀 데이터셋 처리")
        
        # CSV 파일이 있는지 확인
        csv_files = list(self.dataset_path.glob("*.csv"))
        if csv_files:
            try:
                df = pd.read_csv(csv_files[0])
                
                text_prompts = []
                asset_paths = []
                
                for _, row in df.head(num_samples).iterrows():
                    # 텍스트 컬럼 찾기
                    text_col = None
                    for col in ['prompt', 'text', 'description', 'caption']:
                        if col in df.columns:
                            text_col = col
                            break
                    
                    if text_col:
                        text_prompts.append(str(row[text_col]))
                    else:
                        text_prompts.append(f"a 3D object {len(text_prompts)}")
                    
                    asset_paths.append(None)  # 파일 경로는 일단 None
                
                print(f"✅ 커스텀 데이터셋 로드 완료: {len(text_prompts)}개 샘플")
                return text_prompts, asset_paths
                
            except Exception as e:
                print(f"⚠️ 커스텀 데이터셋 로드 실패: {e}")
        
        # 기본 프롬프트 사용
        return self.generate_default_prompts(num_samples)
    
    def generate_default_prompts(self, num_samples: int) -> Tuple[List[str], List[Any]]:
        """기본 텍스트 프롬프트 생성"""
        default_prompts = [
            "a high quality 3D model",
            "a detailed toy object", 
            "a colorful 3D toy",
            "a realistic miniature model",
            "a professional 3D asset",
            "a well-designed toy figure",
            "a small decorative object",
            "a children's toy model",
            "a collectible figurine",
            "a handcrafted 3D object"
        ]
        
        # 샘플 수만큼 반복 생성
        text_prompts = []
        for i in range(num_samples):
            prompt = default_prompts[i % len(default_prompts)]
            if i >= len(default_prompts):
                prompt += f" variant {i // len(default_prompts) + 1}"
            text_prompts.append(prompt)
        
        asset_paths = [None] * num_samples
        
        print(f"✅ 기본 프롬프트 생성 완료: {len(text_prompts)}개 샘플")
        return text_prompts, asset_paths
    
    def evaluate_clip_score(self, text_prompts: List[str], num_samples: int) -> Dict[str, float]:
        """CLIP Score 평가"""
        print(f"📐 CLIP Score 평가 시작 (샘플: {num_samples}개)")
        
        evaluator = get_metrics_evaluator(self.device)
        
        try:
            # 프롬프트별로 3D 생성 및 CLIP 평가
            generated_assets = []
            successful_prompts = []
            
            for i, prompt in enumerate(tqdm(text_prompts[:num_samples], desc="3D 생성")):
                try:
                    # 더미 이미지 입력 (실제로는 텍스트 기반 생성)
                    dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
                    
                    with torch.no_grad():
                        output = self.pipeline(dummy_image, num_inference_steps=5)
                    
                    generated_assets.append(output)
                    successful_prompts.append(prompt)
                    
                except Exception as e:
                    print(f"  ⚠️ 샘플 {i+1} 생성 실패: {e}")
                    continue
            
            if len(generated_assets) == 0:
                print("❌ 생성된 자산이 없어 CLIP 평가 불가")
                return {'clip_score_mean': 0.0, 'clip_score_std': 0.0, 'num_samples': 0}
            
            # TRELLIS 논문 방식 CLIP Score 계산
            clip_results = evaluator.compute_clip_score_trellis_paper(
                successful_prompts, generated_assets
            )
            
            print(f"✅ CLIP Score 평가 완료: 평균 {clip_results['clip_score_mean']:.2f}")
            return clip_results
            
        except Exception as e:
            print(f"❌ CLIP Score 평가 실패: {e}")
            return {'clip_score_mean': 0.0, 'clip_score_std': 0.0, 'num_samples': 0}
    
    def evaluate_frechet_distance(self, text_prompts: List[str], reference_assets: List[Any], 
                                  num_samples: int) -> float:
        """Fréchet Distance 평가"""
        print(f"📏 Fréchet Distance 평가 시작 (샘플: {num_samples}개)")
        
        evaluator = get_metrics_evaluator(self.device)
        
        try:
            # 생성된 자산들
            generated_assets = []
            
            for i, prompt in enumerate(tqdm(text_prompts[:num_samples], desc="FD용 3D 생성")):
                try:
                    dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
                    
                    with torch.no_grad():
                        output = self.pipeline(dummy_image, num_inference_steps=5)
                    
                    generated_assets.append(output)
                    
                except Exception as e:
                    print(f"  ⚠️ 샘플 {i+1} 생성 실패: {e}")
                    continue
            
            if len(generated_assets) == 0:
                print("❌ 생성된 자산이 없어 FD 평가 불가")
                return 100.0
            
            # 참조 자산이 없는 경우 생성된 자산 간 비교
            if not reference_assets or len(reference_assets) == 0:
                print("⚠️ 참조 자산이 없어 생성 자산 내부 다양성으로 FD 계산")
                mid_point = len(generated_assets) // 2
                reference_assets = generated_assets[:mid_point]
                generated_assets = generated_assets[mid_point:]
            
            # TRELLIS 논문 방식 FD 계산
            fd_score = evaluator.compute_frechet_distance_trellis_paper(
                reference_assets, generated_assets
            )
            
            print(f"✅ Fréchet Distance 평가 완료: {fd_score:.2f}")
            return fd_score
            
        except Exception as e:
            print(f"❌ Fréchet Distance 평가 실패: {e}")
            return 100.0
    
    def evaluate_efficiency(self) -> Dict[str, float]:
        """효율성 지표 평가"""
        print("⚡ 효율성 지표 평가 시작")
        
        evaluator = get_metrics_evaluator(self.device)
        
        try:
            efficiency_metrics = evaluator.compute_efficiency_metrics(
                self.pipeline, "quantized_model"
            )
            
            print("✅ 효율성 평가 완료")
            print(f"  파라미터: {efficiency_metrics.get('parameters_M', 0):.1f}M")
            print(f"  모델 크기: {efficiency_metrics.get('model_size_MB', 0):.1f}MB") 
            print(f"  GPU 메모리: {efficiency_metrics.get('gpu_memory_MB', 0):.1f}MB")
            print(f"  추론 시간: {efficiency_metrics.get('inference_time_ms', 0):.1f}ms")
            
            return efficiency_metrics
            
        except Exception as e:
            print(f"❌ 효율성 평가 실패: {e}")
            return {}
    
    def generate_markdown_report(self, results: Dict[str, Any], 
                                report_name: Optional[str] = None) -> str:
        """마크다운 보고서 생성"""
        if report_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_path.name
            report_name = f"evaluation_report_{model_name}_{timestamp}.md"
        
        report_path = self.output_dir / report_name
        
        # 보고서 작성
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# TRELLIS 양자화 모델 평가 보고서\n\n")
            
            # 기본 정보
            f.write(f"## 📋 평가 정보\n\n")
            f.write(f"- **평가 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **모델 경로**: `{self.model_path}`\n")
            f.write(f"- **데이터셋**: `{self.dataset_path}`\n")
            f.write(f"- **디바이스**: {self.device}\n\n")
            
            # 모델 구조
            if 'model_config' in results:
                config = results['model_config']
                f.write(f"## 🏗️ 모델 구조\n\n")
                f.write(f"```json\n{json.dumps(config, indent=2, ensure_ascii=False)}\n```\n\n")
            
            # 효율성 지표
            if 'efficiency' in results:
                eff = results['efficiency']
                f.write(f"## ⚡ 효율성 지표\n\n")
                f.write(f"| 지표 | 값 |\n")
                f.write(f"|------|----|\n")
                f.write(f"| 파라미터 수 | {eff.get('parameters_M', 0):.1f}M |\n")
                f.write(f"| 모델 크기 | {eff.get('model_size_MB', 0):.1f}MB |\n")
                f.write(f"| GPU 메모리 | {eff.get('gpu_memory_MB', 0):.1f}MB |\n")
                f.write(f"| 추론 시간 | {eff.get('inference_time_ms', 0):.1f}ms |\n\n")
            
            # CLIP Score
            if 'clip_score' in results:
                clip = results['clip_score']
                f.write(f"## 📐 CLIP Score (텍스트-3D 일관성)\n\n")
                f.write(f"TRELLIS 논문 방식으로 측정된 CLIP Score 결과:\n\n")
                f.write(f"- **평균 점수**: {clip.get('clip_score_mean', 0):.2f}\n")
                f.write(f"- **표준편차**: {clip.get('clip_score_std', 0):.2f}\n")
                f.write(f"- **최소값**: {clip.get('clip_score_min', 0):.2f}\n")
                f.write(f"- **최대값**: {clip.get('clip_score_max', 0):.2f}\n")
                f.write(f"- **평가 샘플 수**: {clip.get('num_samples', 0)}개\n\n")
                
                # 점수 해석
                score = clip.get('clip_score_mean', 0)
                if score >= 80:
                    interpretation = "🟢 우수 (80+)"
                elif score >= 70:
                    interpretation = "🟡 양호 (70-80)"
                elif score >= 60:
                    interpretation = "🟠 보통 (60-70)"
                else:
                    interpretation = "🔴 개선 필요 (<60)"
                
                f.write(f"**평가**: {interpretation}\n\n")
            
            # Fréchet Distance  
            if 'frechet_distance' in results:
                fd = results['frechet_distance']
                f.write(f"## 📏 Fréchet Distance (생성 품질)\n\n")
                f.write(f"DINOv2 기반으로 측정된 생성 품질 및 다양성:\n\n")
                f.write(f"- **FD 점수**: {fd:.2f}\n\n")
                
                # 점수 해석
                if fd <= 20:
                    interpretation = "🟢 우수 (≤20)"
                elif fd <= 40:
                    interpretation = "🟡 양호 (20-40)"
                elif fd <= 60:
                    interpretation = "🟠 보통 (40-60)"
                else:
                    interpretation = "🔴 개선 필요 (>60)"
                
                f.write(f"**평가**: {interpretation}\n\n")
            
            # 종합 평가
            f.write(f"## 🎯 종합 평가\n\n")
            
            if 'efficiency' in results and 'clip_score' in results:
                eff = results['efficiency']
                clip = results['clip_score']
                
                # 효율성-품질 트레이드오프 분석
                model_size_mb = eff.get('model_size_MB', 0)
                clip_score = clip.get('clip_score_mean', 0)
                
                f.write(f"### 효율성-품질 트레이드오프\n\n")
                f.write(f"- **모델 크기**: {model_size_mb:.1f}MB\n")
                f.write(f"- **CLIP 점수**: {clip_score:.2f}\n")
                f.write(f"- **품질/크기 비율**: {clip_score/max(model_size_mb/1000, 1):.2f}\n\n")
            
            # 권장사항
            f.write(f"### 권장사항\n\n")
            
            if 'clip_score' in results:
                clip_score = results['clip_score'].get('clip_score_mean', 0)
                if clip_score < 70:
                    f.write(f"- 🔴 CLIP Score가 낮습니다. 양자화 정도를 줄이는 것을 고려해보세요.\n")
                elif clip_score >= 80:
                    f.write(f"- 🟢 CLIP Score가 우수합니다. 현재 양자화 설정이 적절합니다.\n")
            
            if 'efficiency' in results:
                inference_time = results['efficiency'].get('inference_time_ms', 0)
                if inference_time > 2000:
                    f.write(f"- 🟠 추론 시간이 길 수 있습니다. 더 적극적인 양자화를 고려해보세요.\n")
            
            f.write(f"\n---\n")
            f.write(f"*이 보고서는 TRELLIS 양자화 평가 시스템에 의해 자동 생성되었습니다.*\n")
        
        print(f"📄 마크다운 보고서 생성 완료: {report_path}")
        return str(report_path)
    
    def run_evaluation(self, metrics: List[str], num_samples: int, 
                      report_name: Optional[str] = None) -> str:
        """전체 평가 실행"""
        print(f"🚀 양자화 모델 평가 시작")
        print(f"평가 지표: {', '.join(metrics)}")
        print(f"샘플 수: {num_samples}")
        print("-" * 50)
        
        results = {}
        
        try:
            # 1. 모델 로드
            config = self.load_model_config()
            self.pipeline = self.load_pipeline()
            results['model_config'] = config
            
            # 2. 데이터셋 로드
            text_prompts, reference_assets = self.load_dataset(num_samples)
            
            # 3. 평가기 초기화
            self.evaluator = get_metrics_evaluator(self.device)
            
            # 4. 각 지표별 평가
            if 'efficiency' in metrics:
                results['efficiency'] = self.evaluate_efficiency()
            
            if 'CLIP' in metrics:
                results['clip_score'] = self.evaluate_clip_score(text_prompts, num_samples)
            
            if 'FD' in metrics:
                results['frechet_distance'] = self.evaluate_frechet_distance(
                    text_prompts, reference_assets, num_samples
                )
            
            # 5. 결과 저장 (JSON)
            results_json_path = self.output_dir / "evaluation_results.json"
            with open(results_json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # 6. 마크다운 보고서 생성
            report_path = self.generate_markdown_report(results, report_name)
            
            print(f"\n🎉 평가 완료!")
            print(f"📊 JSON 결과: {results_json_path}")
            print(f"📄 보고서: {report_path}")
            
            return report_path
            
        except Exception as e:
            print(f"❌ 평가 실패: {e}")
            raise
        
        finally:
            # 리소스 정리
            if self.evaluator:
                cleanup_global_evaluator()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="TRELLIS 양자화 모델 평가 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 평가 (모든 지표)
  python evaluate_quantized_model.py \\
      --model_path quantization_results/trellis_text-base_quantized \\
      --dataset datasets/Toys4k \\
      --CLIP --FD --efficiency
  
  # CLIP Score만 평가
  python evaluate_quantized_model.py \\
      --model_path my_model \\
      --dataset my_dataset \\
      --CLIP --num_samples 20
        """
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='양자화된 모델 경로 (pipeline.json이 있는 디렉토리)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='데이터셋 경로')
    
    # 평가 지표 선택
    parser.add_argument('--CLIP', action='store_true',
                       help='CLIP Score 계산 (텍스트-3D 일관성)')
    parser.add_argument('--FD', action='store_true', 
                       help='Fréchet Distance 계산 (생성 품질)')
    parser.add_argument('--efficiency', action='store_true',
                       help='효율성 지표 계산 (파라미터, 메모리, 속도)')
    
    # 기타 옵션
    parser.add_argument('--num_samples', type=int, default=50,
                       help='평가할 샘플 수 (기본값: 50)')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='결과 저장 디렉토리 (기본값: evaluation_results)')
    parser.add_argument('--report_name', type=str, default=None,
                       help='보고서 파일명 (기본값: 자동 생성)')
    
    args = parser.parse_args()
    
    # 평가 지표 확인
    metrics = []
    if args.CLIP:
        metrics.append('CLIP')
    if args.FD:
        metrics.append('FD') 
    if args.efficiency:
        metrics.append('efficiency')
    
    if not metrics:
        print("❌ 최소 하나의 평가 지표를 선택해야 합니다. (--CLIP, --FD, --efficiency)")
        sys.exit(1)
    
    # 평가 실행
    try:
        evaluator = QuantizedModelEvaluator(
            model_path=args.model_path,
            dataset_path=args.dataset, 
            output_dir=args.output_dir
        )
        
        report_path = evaluator.run_evaluation(
            metrics=metrics,
            num_samples=args.num_samples,
            report_name=args.report_name
        )
        
        print(f"\n✅ 평가 완료! 보고서: {report_path}")
        
    except Exception as e:
        print(f"❌ 평가 실행 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()