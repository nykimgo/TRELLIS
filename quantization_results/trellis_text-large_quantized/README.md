# TRELLIS TEXT-LARGE Quantized Model (INT8)

## 📋 모델 정보
- **원본 모델**: /home/sr/TRELLIS/microsoft/TRELLIS-text-large
- **양자화 방법**: Dynamic INT8 Quantization
- **양자화 일시**: 2025-08-05 10:06:06
- **저장된 컴포넌트**: 6개

## 🚀 사용 방법

```python
from trellis.pipelines import TrellisTextTo3DPipeline

# 양자화된 모델 로드
pipeline = TrellisTextTo3DPipeline.from_pretrained("quantization_results/trellis_text-large_quantized")
pipeline.cuda()

# 3D 생성
outputs = pipeline.run("a red sports car", seed=42)

# 결과 저장
from trellis.utils import postprocessing_utils
glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0])
glb.export("output.glb")
```

## 📊 저장된 컴포넌트
- sparse_structure_decoder: 140.8MB
- sparse_structure_flow_model: 2086.4MB
- slat_decoder_gs: 217.5MB
- slat_decoder_rf: 217.5MB
- slat_decoder_mesh: 227.5MB
- slat_flow_model: 1777.5MB

## ⚠️ 주의사항
- INT8 양자화로 인해 원본 대비 약간의 품질 저하가 있을 수 있습니다
- GPU 메모리 사용량과 추론 속도는 개선됩니다

## 🔧 파일 구조
- `pipeline.json`: 파이프라인 설정
- `ckpts/*.json`: 각 컴포넌트의 모델 설정
- `ckpts/*.safetensors`: 양자화된 모델 가중치
- `README.md`: 사용 가이드
