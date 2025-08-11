"""
TRELLIS 양자화 관리 클래스

주요 기능:
- 모델 로드 및 구조 분석
- Dynamic INT8 양자화 적용
- 성능 측정 및 비교
"""

import os
import torch
import torch.nn as nn
import time
import gc
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# 상대 임포트 문제 해결
try:
    from model_analyzer import ModelAnalyzer
    from performance_measurer import PerformanceMeasurer
    from model_saver import ModelSaver
except ImportError:
    # 절대 임포트로 시도
    from .model_analyzer import ModelAnalyzer
    from .performance_measurer import PerformanceMeasurer
    from .model_saver import ModelSaver


class ActivationQuantWrapper(nn.Module):
    """간단한 대칭 활성화 양자화 래퍼"""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    @staticmethod
    def _quant_tensor(t: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(t):
            return t
        scale = t.abs().max() / 127 if t.numel() > 0 else 1.0
        if scale == 0:
            return t
        q = torch.clamp((t / scale).round(), -128, 127).to(torch.int8)
        dq = q.to(torch.float32) * scale
        return dq

    def forward(self, *args, **kwargs):
        q_args = [self._quant_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
        q_kwargs = {k: self._quant_tensor(v) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        out = self.module(*q_args, **q_kwargs)
        if isinstance(out, torch.Tensor):
            return self._quant_tensor(out)
        if isinstance(out, (list, tuple)):
            return type(out)(self._quant_tensor(o) if isinstance(o, torch.Tensor) else o for o in out)


class TRELLISQuantizationManager:
    """TRELLIS 양자화 관리 클래스"""
    
    # 지원되는 TRELLIS 모델들
    SUPPORTED_MODELS = {
        "text-base": "/home/sr/TRELLIS/microsoft/TRELLIS-text-base",
        "text-large": "/home/sr/TRELLIS/microsoft/TRELLIS-text-large", 
        "text-xlarge": "/home/sr/TRELLIS/microsoft/TRELLIS-text-xlarge",
        "image-large": "/home/sr/TRELLIS/microsoft/TRELLIS-image-large"
    }
    
    # 양자화 대상 레이어 타입
    QUANTIZABLE_LAYERS = {
        nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, 
        nn.ConvTranspose3d, nn.MultiheadAttention
    }
    
    def __init__(
        self,
        model_name: str = "text-base",
        output_dir: str = "quantization_results",
        modules_to_quantize: Optional[List[str]] = None,
        quantize_weights: bool = True,
        quantize_activations: bool = False,
    ):
        """초기화

        Args:
            model_name: 사용할 TRELLIS 모델 이름
            output_dir: 결과 저장 디렉토리
            modules_to_quantize: 양자화할 모듈 이름 리스트. None이면 전체 모듈 대상
            quantize_weights: 가중치 INT8 양자화 여부
            quantize_activations: 활성화 INT8 양자화 여부
        """
        self.model_name = model_name
        self.model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 양자화 설정
        self.modules_to_quantize = modules_to_quantize
        self.quantize_weights = quantize_weights
        self.quantize_activations = quantize_activations

        
        # 상태 변수
        self.original_pipeline = None
        self.quantized_pipeline = None
        self.model_components = []  # (name, module) 튜플 리스트
        self.results = []
        
        # 헬퍼 클래스들
        self.analyzer = ModelAnalyzer()
        self.measurer = PerformanceMeasurer()
        self.saver = ModelSaver(self.output_dir)
        
        # 환경 설정
        os.environ['SPCONV_ALGO'] = 'native'
        os.environ['ATTN_BACKEND'] = 'xformers'
        
        print(f"🔧 TRELLIS 양자화 매니저 초기화")
        print(f"  📂 모델: {self.model_path}")
        print(f"  📁 출력: {self.output_dir}")

        if self.modules_to_quantize:
            print(f"  🎯 대상 모듈: {self.modules_to_quantize}")
        print(f"  ⚙️ 가중치 양자화: {self.quantize_weights}, 활성화 양자화: {self.quantize_activations}")
    
    def load_original_model(self) -> bool:
        """원본 모델 로드"""
        try:
            print(f"🔄 TRELLIS 파이프라인 로드 중...")
            
            # 경로 확인
            if not os.path.exists(self.model_path):
                print(f"❌ 모델 경로 없음: {self.model_path}")
                self._show_available_models()
                return False
            
            # TRELLIS 모듈 임포트
            try:
                if "text" in self.model_name:
                    from trellis.pipelines import TrellisTextTo3DPipeline
                    pipeline_class = TrellisTextTo3DPipeline
                else:
                    from trellis.pipelines import TrellisImageTo3DPipeline  
                    pipeline_class = TrellisImageTo3DPipeline
            except ImportError as e:
                print(f"❌ TRELLIS 모듈 임포트 실패: {e}")
                print("💡 TRELLIS 프로젝트 루트에서 실행하거나 PYTHONPATH 설정")
                return False
            
            # 파이프라인 로드
            self.original_pipeline = pipeline_class.from_pretrained(self.model_path)
            
            # GPU로 이동
            if torch.cuda.is_available():
                try:
                    self.original_pipeline.cuda()
                    print("✅ 모델을 GPU로 로드")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("⚠️ GPU 메모리 부족 - CPU 사용")
                        torch.cuda.empty_cache()
                    else:
                        raise e
            
            # 모델 구조 분석
            self.model_components = self.analyzer.analyze_pipeline(self.original_pipeline)

            # 특정 모듈만 선택적으로 양자화
            if self.modules_to_quantize:
                self.model_components = [
                    (n, m) for n, m in self.model_components if n in self.modules_to_quantize
                ]
                print(f"🎯 선택된 모듈 수: {len(self.model_components)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def apply_quantization(self) -> bool:
        """양자화 적용"""
        try:
            print("🔧 INT8 양자화 적용 중...")
            
            if not self.model_components:
                print("❌ 양자화할 컴포넌트가 없습니다")
                return False
            
            # 메모리 정리
            torch.cuda.empty_cache()
            gc.collect()
            
            success_count = 0
            total_count = len(self.model_components)
            
            for name, original_module in self.model_components:
                success = self._quantize_component(name, original_module)
                if success:
                    success_count += 1
            
            # 양자화된 파이프라인 설정 (참조 복사)
            self.quantized_pipeline = self.original_pipeline
            
            torch.cuda.empty_cache()
            gc.collect()
            
            print(f"📊 양자화 결과: {success_count}/{total_count} 성공")
            return success_count > 0
            
        except Exception as e:
            print(f"❌ 양자화 실패: {e}")
            return False
    
    def _quantize_component(self, name: str, module: nn.Module) -> bool:
        """개별 컴포넌트 양자화"""
        try:
            print(f"  🔧 {name} 양자화 중...")
            
            # 파라미터 확인
            original_params = sum(p.numel() for p in module.parameters())
            if original_params == 0:
                print(f"    ⚠️ 파라미터 없음 - 건너뜀")
                return False
            
            # CPU로 이동 후 필요 시 가중치 양자화
            module.cpu()
            torch.cuda.empty_cache()
            
            quantized_module = module
            size_reduction = 0.0

            if self.quantize_weights:
                quantized_module = torch.quantization.quantize_dynamic(
                    module,
                    self.QUANTIZABLE_LAYERS,
                    dtype=torch.qint8
                )
                original_size = sum(p.numel() * p.element_size() for p in module.parameters())
                quantized_size = sum(p.numel() * p.element_size() for p in quantized_module.parameters())
                if original_size > 0:
                    size_reduction = ((original_size - quantized_size) / original_size) * 100

            # 활성화 양자화 래핑
            if self.quantize_activations:
                quantized_module = ActivationQuantWrapper(quantized_module)

            # GPU로 복귀 및 교체
            if torch.cuda.is_available():
                quantized_module.cuda()
            
            if hasattr(self.original_pipeline, 'models') and name in self.original_pipeline.models:
                self.original_pipeline.models[name] = quantized_module
            elif hasattr(self.original_pipeline, name):
                setattr(self.original_pipeline, name, quantized_module)
            else:
                print(f"    ⚠️ 컴포넌트 교체 실패")
                return False
            
            print(f"    ✅ 완료 - 크기 감소: {size_reduction:.1f}%")
            return True
            
        except Exception as e:
            print(f"    ❌ 실패: {e}")
            return False
    
    def run_experiment(self) -> bool:
        """전체 실험 실행"""
        try:
            print("🚀 TRELLIS 양자화 실험 시작")
            print("=" * 50)
            
            # 1. 원본 모델 로드
            if not self.load_original_model():
                return False
            
            # 2. 원본 성능 측정
            original_metrics = self.measurer.measure_performance(
                self.original_pipeline, "Original (FP32)", self.model_components
            )
            self.results.append(original_metrics)
            
            # 3. 양자화 적용
            if not self.apply_quantization():
                return False
            
            # 4. 양자화 성능 측정
            quantized_metrics = self.measurer.measure_performance(
                self.quantized_pipeline, "Quantized (INT8)", self.model_components
            )
            self.results.append(quantized_metrics)
            
            # 5. 결과 저장
            self.saver.save_results(self.results, self.model_name)
            
            # 6. 양자화 모델 저장
            quantized_path = self.saver.save_quantized_model(
                self.quantized_pipeline, self.model_name, self.model_path
            )
            
            print("\n✅ 실험 완료!")
            if quantized_path:
                print(f"💾 양자화 모델: {quantized_path}")
            print(f"📊 결과 확인: {self.output_dir}/")
            
            return True
            
        except Exception as e:
            print(f"❌ 실험 실행 오류: {e}")
            return False
    
    def _show_available_models(self):
        """사용 가능한 모델 목록 표시"""
        print("💡 사용 가능한 모델들:")
        for name, path in self.SUPPORTED_MODELS.items():
            exists = "✅" if os.path.exists(path) else "❌"
            print(f"  {exists} {name}: {path}")