
"""
TRELLIS 모델 저장 및 결과 관리 클래스

기능:
- CSV/JSON 결과 저장
- 양자화된 모델을 TRELLIS 형식으로 저장
- 시각화 그래프 생성
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import torch

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
            
            # 모델 컴포넌트들을 safetensors로 저장
            if hasattr(pipeline, 'models') and pipeline.models:
                for comp_name, model in pipeline.models.items():
                    if model is not None:
                        success = self._save_component(comp_name, model, ckpts_dir)
                        if success:
                            file_path = ckpts_dir / f"{comp_name}_quantized_int8.safetensors"
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            saved_components.append(f"{comp_name}: {file_size_mb:.1f}MB")
            
            # pipeline.json 생성
            self._create_pipeline_config(save_dir, original_path, saved_components)
            
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
    
    def _save_component(self, name: str, model, ckpts_dir: Path) -> bool:
        """개별 컴포넌트 저장"""
        try:
            model_path = ckpts_dir / f"{name}_quantized_int8.safetensors"
            
            # CPU로 이동하여 저장
            model_cpu = model.cpu()
            state_dict = model_cpu.state_dict()
            torch.save(state_dict, model_path)
            
            # GPU로 복귀
            if torch.cuda.is_available():
                model.cuda()
            
            return True
            
        except Exception as e:
            print(f"    ❌ {name} 저장 실패: {e}")
            return False
    
    def _create_pipeline_config(self, save_dir: Path, original_path: str, saved_components: List[str]):
        """pipeline.json 설정 파일 생성"""
        try:
            # 원본 설정 읽기
            original_config_path = Path(original_path) / "pipeline.json"
            if original_config_path.exists():
                with open(original_config_path, 'r') as f:
                    config = json.load(f)
                
                # 모델 경로를 양자화 버전으로 수정
                if 'args' in config and 'models' in config['args']:
                    for model_name in config['args']['models']:
                        config['args']['models'][model_name] = f"ckpts/{model_name}_quantized_int8.safetensors"
                
                # 양자화 정보 추가
                config['quantization_info'] = {
                    'method': 'dynamic_int8',
                    'original_model': original_path,
                    'quantized_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'saved_components': [comp.split(':')[0] for comp in saved_components]
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