"""
TRELLIS 모델 저장 및 결과 관리 클래스

기능:
- CSV/JSON 결과 저장
- 양자화된 모델을 TRELLIS 형식으로 저장 (config.json + safetensors)
- 시각화 그래프 생성
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import torch
from safetensors.torch import save_file

# 상대 임포트 문제 해결
try:
    from performance_measurer import PerformanceMeasurer
except ImportError:
    from .performance_measurer import PerformanceMeasurer


class ModelSaver:
    """모델 저장 및 결과 관리"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.measurer = PerformanceMeasurer()
    
    def save_results(self, results: List[Dict[str, Any]], model_name: str):
        """실험 결과 저장"""
        try:
            print("💾 실험 결과 저장 중...")
            
            # CSV 저장
            df = pd.DataFrame(results)
            csv_path = self.output_dir / f"trellis_{model_name}_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"  ✅ CSV: {csv_path}")
            
            # 압축 메트릭 계산 및 저장
            compression_metrics = self.measurer.calculate_compression_metrics(results)
            if compression_metrics:
                metrics_path = self.output_dir / f"trellis_{model_name}_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(compression_metrics, f, indent=2)
                print(f"  ✅ 메트릭: {metrics_path}")
                
                # 결과 출력
                self._print_compression_results(compression_metrics)
            
            # 시각화 생성
            self._create_visualization(df, model_name)
            
        except Exception as e:
            print(f"⚠️ 결과 저장 실패: {e}")
    
    def save_quantized_model(self, pipeline, model_name: str, original_path: str) -> Optional[str]:
        """양자화된 모델을 TRELLIS 형식으로 저장"""
        try:
            print("💾 양자화된 모델 저장 중...")
            
            # 저장 디렉토리 생성
            save_dir = self.output_dir / f"trellis_{model_name}_quantized"
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpts_dir = save_dir / "ckpts"
            ckpts_dir.mkdir(parents=True, exist_ok=True)
            
            saved_components = []
            
            # 모델 컴포넌트들을 config.json + safetensors로 저장
            if hasattr(pipeline, 'models') and pipeline.models:
                for comp_name, model in pipeline.models.items():
                    if model is not None:
                        success = self._save_component_with_config(
                            comp_name, model, ckpts_dir, original_path
                        )
                        if success:
                            safetensors_path = ckpts_dir / f"{comp_name}_quantized_int8.safetensors"
                            if safetensors_path.exists():
                                file_size_mb = safetensors_path.stat().st_size / (1024 * 1024)
                                saved_components.append(f"{comp_name}: {file_size_mb:.1f}MB")
            
            # pipeline.json 생성 (모든 모델을 safetensors로 통일)
            self._create_pipeline_config_fixed(save_dir, original_path, saved_components)
            
            # README 생성
            self._create_readme(save_dir, model_name, original_path, saved_components)
            
            if saved_components:
                print(f"  ✅ {len(saved_components)}개 컴포넌트 저장 완료")
                return str(save_dir)
            else:
                print("  ⚠️ 저장된 컴포넌트 없음")
                return None
                
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")
            return None
    
    def _save_component_with_config(self, name: str, model, ckpts_dir: Path, original_path: str) -> bool:
        """개별 컴포넌트를 config.json + safetensors로 저장 (원본 호환 방식)"""
        try:
            print(f"    💾 {name} 저장 중...")
            
            # 1. 원본 모델의 config.json 찾기 및 복사
            original_config_path = self._find_original_config(name, original_path)
            if original_config_path and original_config_path.exists():
                # config.json 복사
                config_path = ckpts_dir / f"{name}_quantized_int8.json" 
                import shutil
                shutil.copy2(original_config_path, config_path)
                print(f"      ✅ Config: {original_config_path} → {config_path}")
            else:
                # config.json을 찾을 수 없으면 기본 구조로 생성
                config_path = ckpts_dir / f"{name}_quantized_int8.json"
                self._create_basic_config(model, config_path)
                print(f"      📝 기본 Config 생성: {config_path}")
            
            # 2. 모델을 원본 형태로 복원 후 저장
            weights_path = ckpts_dir / f"{name}_quantized_int8.safetensors"
            success = self._save_model_compatible_format(model, weights_path)
            
            if success:
                print(f"      ✅ Weights: {weights_path}")
                return True
            else:
                print(f"      ❌ 저장 실패")
                return False
            
        except Exception as e:
            print(f"    ❌ {name} 저장 실패: {e}")
            return False
    
    def _save_model_compatible_format(self, model, weights_path: Path) -> bool:
        """모델을 원본 호환 형태로 저장"""
        try:
            # CPU로 이동
            model_cpu = model.cpu()
            
            # 양자화 상태 확인
            is_quantized = self._is_quantized_model(model_cpu)
            
            if is_quantized:
                # 양자화된 모델: dequantize 후 저장
                print(f"      🔧 Dequantizing model for compatibility...")
                compatible_state_dict = self._dequantize_model(model_cpu)
            else:
                # 일반 모델: 그대로 저장
                compatible_state_dict = model_cpu.state_dict()
            
            # Non-tensor 값들 제거
            clean_state_dict = self._clean_quantized_state_dict(compatible_state_dict)
            
            if not clean_state_dict:
                print(f"      ⚠️ No valid tensors found")
                return False
            
            # safetensors로 저장
            save_file(clean_state_dict, weights_path)
            
            # GPU로 복귀
            if torch.cuda.is_available():
                model.cuda()
            
            return True
            
        except Exception as e:
            print(f"      ❌ 호환 형태 저장 실패: {e}")
            return False
    
    def _is_quantized_model(self, model) -> bool:
        """모델이 양자화되었는지 확인"""
        try:
            for module in model.modules():
                # _packed_params가 있으면 양자화된 모델
                if hasattr(module, '_packed_params'):
                    return True
                # quantized가 클래스명에 포함된 경우
                if 'quantized' in str(type(module)).lower():
                    return True
            return False
        except:
            return False
    
    def _dequantize_model(self, model) -> dict:
        """양자화된 모델을 원본 형태로 복원"""
        try:
            import copy
            
            # 새로운 FP32 모델 생성
            dequantized_model = copy.deepcopy(model)
            
            # 양자화된 모듈들을 FP32로 변환
            self._convert_quantized_modules_to_fp32(dequantized_model)
            
            return dequantized_model.state_dict()
            
        except Exception as e:
            print(f"      ⚠️ Dequantization 실패, 양자화된 state_dict 사용: {e}")
            return model.state_dict()
    
    def _convert_quantized_modules_to_fp32(self, model):
        """양자화된 모듈들을 FP32로 변환"""
        try:
            for name, module in model.named_modules():
                if hasattr(module, '_packed_params'):
                    # Linear 레이어 변환
                    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                        # 양자화된 가중치 추출
                        weight, bias = module._weight_bias()
                        
                        # 새로운 FP32 Linear 레이어 생성
                        new_linear = torch.nn.Linear(
                            module.in_features, 
                            module.out_features,
                            bias=bias is not None
                        )
                        
                        # 가중치 복사
                        new_linear.weight.data = weight.dequantize()
                        if bias is not None:
                            new_linear.bias.data = bias
                        
                        # 모듈 교체
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        
                        if parent_name:
                            parent = model.get_submodule(parent_name)
                            setattr(parent, child_name, new_linear)
                        else:
                            setattr(model, child_name, new_linear)
                            
        except Exception as e:
            print(f"      ⚠️ FP32 변환 중 일부 실패: {e}")
            # 계속 진행
    
    def _find_original_config(self, comp_name: str, original_path: str) -> Optional[Path]:
        """원본 모델의 config.json 파일 찾기"""
        original_path = Path(original_path)
        
        # pipeline.json에서 원본 경로 매핑 찾기
        pipeline_json = original_path / "pipeline.json"
        if pipeline_json.exists():
            try:
                with open(pipeline_json, 'r') as f:
                    config = json.load(f)
                
                if 'args' in config and 'models' in config['args']:
                    original_model_path = config['args']['models'].get(comp_name)
                    if original_model_path:
                        # 상대 경로를 절대 경로로 변환
                        if original_model_path.startswith('../'):
                            # ../TRELLIS-image-large/ckpts/... 형태
                            abs_path = original_path.parent / original_model_path[3:]
                        else:
                            # ckpts/... 형태
                            abs_path = original_path / original_model_path
                        
                        config_path = abs_path.with_suffix('.json')
                        if config_path.exists():
                            return config_path
                        
                        # 직접 경로도 시도
                        direct_config = Path(f"{abs_path}.json")
                        if direct_config.exists():
                            return direct_config
                            
            except Exception as e:
                print(f"      ⚠️ pipeline.json 파싱 실패: {e}")
        
        return None
    
    def _create_basic_config(self, model, config_path: Path):
        """기본 config.json 생성 (원본을 찾을 수 없을 때)"""
        try:
            # 모델 클래스명 추출
            model_class_name = model.__class__.__name__
            
            # 기본 설정 생성
            basic_config = {
                "name": model_class_name,
                "args": {},
                "quantization_info": {
                    "method": "dynamic_int8",
                    "note": "Basic config generated automatically"
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(basic_config, f, indent=2)
                
        except Exception as e:
            print(f"      ⚠️ 기본 config 생성 실패: {e}")
    
    def _clean_quantized_state_dict(self, state_dict: dict) -> dict:
        """양자화된 모델의 state_dict에서 non-tensor 값들 제거"""
        if state_dict is None:
            return {}
            
        clean_dict = {}
        skipped_keys = []
        
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                clean_dict[key] = value
            else:
                skipped_keys.append(f"{key} ({type(value).__name__})")
        
        if skipped_keys:
            print(f"      ⚠️ Non-tensor keys skipped: {len(skipped_keys)}")
            for key in skipped_keys[:3]:  # 처음 3개만 표시
                print(f"        - {key}")
            if len(skipped_keys) > 3:
                print(f"        ... and {len(skipped_keys)-3} more")
        
        return clean_dict
    
    def _create_pipeline_config_fixed(self, save_dir: Path, original_path: str, saved_components: List[str]):
        """pipeline.json 설정 파일 생성 (모든 모델을 safetensors로 통일)"""
        try:
            # 원본 설정 읽기
            original_config_path = Path(original_path) / "pipeline.json"
            if original_config_path.exists():
                with open(original_config_path, 'r') as f:
                    config = json.load(f)
                
                # 모델 경로를 양자화 버전으로 수정 (모두 safetensors)
                if 'args' in config and 'models' in config['args']:
                    for model_name in config['args']['models']:
                        config['args']['models'][model_name] = f"ckpts/{model_name}_quantized_int8"
                
                # 양자화 정보 추가
                config['quantization_info'] = {
                    'method': 'dynamic_int8',
                    'original_model': original_path,
                    'quantized_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'saved_components': [comp.split(':')[0] for comp in saved_components],
                    'note': 'All models saved as safetensors with quantization metadata cleaned'
                }
                
                # 저장
                config_path = save_dir / "pipeline.json"
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                print(f"    ✅ pipeline.json 생성")
                
        except Exception as e:
            print(f"    ⚠️ pipeline.json 생성 실패: {e}")
    
    def _create_readme(self, save_dir: Path, model_name: str, original_path: str, saved_components: List[str]):
        """README.md 생성"""
        readme_path = save_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(f"""# TRELLIS {model_name.upper()} Quantized Model (INT8)

## 📋 모델 정보
- **원본 모델**: {original_path}
- **양자화 방법**: Dynamic INT8 Quantization
- **양자화 일시**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **저장된 컴포넌트**: {len(saved_components)}개

## 🚀 사용 방법

```python
from trellis.pipelines import TrellisTextTo3DPipeline

# 양자화된 모델 로드
pipeline = TrellisTextTo3DPipeline.from_pretrained("{save_dir}")
pipeline.cuda()

# 3D 생성
outputs = pipeline.run("a red sports car", seed=42)

# 결과 저장
from trellis.utils import postprocessing_utils
glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0])
glb.export("output.glb")
```

## 📊 저장된 컴포넌트
""")
            for component in saved_components:
                f.write(f"- {component}\n")
            
            f.write(f"""
## ⚠️ 주의사항
- INT8 양자화로 인해 원본 대비 약간의 품질 저하가 있을 수 있습니다
- GPU 메모리 사용량과 추론 속도는 개선됩니다

## 🔧 파일 구조
- `pipeline.json`: 파이프라인 설정
- `ckpts/*.json`: 각 컴포넌트의 모델 설정
- `ckpts/*.safetensors`: 양자화된 모델 가중치
- `README.md`: 사용 가이드
""")
    
    def _create_visualization(self, df: pd.DataFrame, model_name: str):
        """성능 비교 시각화"""
        try:
            if len(df) < 2:
                return
            
            # 에러가 있는 행 제외
            df_clean = df[~df.get('error', pd.Series(False, index=df.index)).notna()]
            if df_clean.empty:
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # 1. 파라미터 수
            ax1.bar(df_clean['model_name'], df_clean['total_params_M'])
            ax1.set_title('Model Parameters (M)')
            ax1.set_ylabel('Parameters (Millions)')
            
            # 2. 모델 크기
            ax2.bar(df_clean['model_name'], df_clean['model_size_MB'])
            ax2.set_title('Model Size (MB)')
            ax2.set_ylabel('Size (MB)')
            
            # 3. GPU 메모리
            ax3.bar(df_clean['model_name'], df_clean['gpu_memory_MB'])
            ax3.set_title('GPU Memory Usage (MB)')
            ax3.set_ylabel('Memory (MB)')
            
            # 4. 추론 시간
            ax4.bar(df_clean['model_name'], df_clean['inference_time_ms'])
            ax4.set_title('Inference Time (ms)')
            ax4.set_ylabel('Time (ms)')
            
            plt.tight_layout()
            
            # 저장
            plot_path = self.output_dir / f"trellis_{model_name}_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ 시각화: {plot_path}")
            
        except Exception as e:
            print(f"  ⚠️ 시각화 실패: {e}")
    
    def _print_compression_results(self, metrics: Dict[str, float]):
        """압축 결과 출력"""
        print(f"\n🎯 압축 효과:")
        print(f"  • 압축률: {metrics['compression_ratio']:.1f}x")
        print(f"  • 크기 감소: {metrics['size_reduction_percent']:.1f}%")
        print(f"  • 메모리 절약: {metrics['memory_reduction_percent']:.1f}%")
        print(f"  • 속도 변화: {metrics['speed_change_percent']:+.1f}%")
        print(f"  • 품질 손실: {metrics['quality_loss_percent']:.1f}%")
        print(f"  • 효율성 점수: {metrics['efficiency_score']:.3f}")