# 🎯 TRELLIS 양자화 모델 평가 가이드

미리 양자화해둔 체크포인트들을 평가하는 도구입니다.

## 🚀 빠른 시작

### 기본 사용법

```bash
python evaluate_quantized_model.py \
    --model_path quantization_results/trellis_text-base_quantized \
    --dataset datasets/Toys4k \
    --CLIP --FD --efficiency \
    --num_samples 50
```

### 필수 인자

- `--model_path`: 양자화된 모델이 저장된 디렉토리 경로 (pipeline.json 파일이 있어야 함)
- `--dataset`: 데이터셋 경로

### 평가 지표 선택

- `--CLIP`: CLIP Score 계산 (텍스트-3D 일관성)
- `--FD`: Fréchet Distance 계산 (생성 품질)  
- `--efficiency`: 효율성 지표 계산 (파라미터, 메모리, 속도)

### 선택적 인자

- `--num_samples`: 평가할 샘플 수 (기본값: 50)
- `--output_dir`: 결과 저장 디렉토리 (기본값: evaluation_results)
- `--report_name`: 보고서 파일명 (기본값: 자동 생성)

## 📁 디렉토리 구조

양자화된 모델 디렉토리는 다음 구조를 가져야 합니다:

```
quantization_results/trellis_text-base_quantized/
├── pipeline.json          # 파이프라인 설정 파일
├── model_index.json       # 모델 인덱스 (선택사항)
├── G_L/                   # G_L 모델 체크포인트
│   ├── config.json
│   └── pytorch_model.bin
├── G_S/                   # G_S 모델 체크포인트  
│   ├── config.json
│   └── pytorch_model.bin
└── ...                    # 기타 모델 파일들
```

## 🎯 사용 예시

### 1. 전체 평가 (모든 지표)

```bash
python evaluate_quantized_model.py \
    --model_path quantization_results/trellis_text-base_quantized \
    --dataset datasets/Toys4k \
    --CLIP --FD --efficiency \
    --num_samples 100 \
    --output_dir my_evaluation_results
```

### 2. CLIP Score만 빠르게 테스트

```bash
python evaluate_quantized_model.py \
    --model_path my_quantized_model \
    --dataset datasets/Toys4k \
    --CLIP \
    --num_samples 20
```

### 3. 효율성 지표만 측정

```bash
python evaluate_quantized_model.py \
    --model_path my_model \
    --dataset datasets/Toys4k \
    --efficiency
```

### 4. 커스텀 데이터셋 사용

```bash
python evaluate_quantized_model.py \
    --model_path my_model \
    --dataset /path/to/my/custom/dataset \
    --CLIP --FD \
    --num_samples 30 \
    --report_name my_custom_evaluation
```

## 📊 출력 결과

### 1. JSON 결과 파일
`evaluation_results/evaluation_results.json`
```json
{
  "model_config": {...},
  "efficiency": {
    "parameters_M": 125.3,
    "model_size_MB": 2140.5,
    "gpu_memory_MB": 8192.0,
    "inference_time_ms": 850.2
  },
  "clip_score": {
    "clip_score_mean": 78.5,
    "clip_score_std": 5.2,
    "clip_score_min": 65.1,
    "clip_score_max": 89.3,
    "num_samples": 50
  },
  "frechet_distance": 32.1
}
```

### 2. 마크다운 보고서
`evaluation_results/evaluation_report_[model_name]_[timestamp].md`

보고서에는 다음 내용이 포함됩니다:
- 📋 평가 정보 (일시, 모델, 데이터셋)
- 🏗️ 모델 구조 정보
- ⚡ 효율성 지표 테이블
- 📐 CLIP Score 상세 결과
- 📏 Fréchet Distance 결과
- 🎯 종합 평가 및 권장사항

## 🔧 지원 데이터셋

### Toys4k 데이터셋
```bash
# Toys4k 메타데이터 자동 로드
--dataset datasets/Toys4k
```

### 커스텀 데이터셋
CSV 파일이 있는 경우 자동으로 텍스트 컬럼을 찾아 프롬프트로 사용:
- 지원 컬럼명: `prompt`, `text`, `description`, `caption`

### 기본 프롬프트
데이터셋을 찾을 수 없는 경우 기본 프롬프트 사용:
- "a high quality 3D model"
- "a detailed toy object"  
- "a colorful 3D toy"
- 등...

## ⚡ 성능 최적화

### GPU 메모리 절약
```bash
# 샘플 수를 줄여서 메모리 사용량 감소
--num_samples 20

# 효율성 지표만 측정 (3D 생성 없음)
--efficiency
```

### 빠른 테스트
```bash
# CLIP Score만 측정하여 빠른 품질 확인
--CLIP --num_samples 10
```

## 🐛 트러블슈팅

### 1. 모델 로드 실패
```
❌ 파이프라인 로드 실패: FileNotFoundError
```
→ `pipeline.json` 파일이 모델 디렉토리에 있는지 확인

### 2. 데이터셋 로드 실패
```
⚠️ Toys4k 데이터셋 로드 실패: ModuleNotFoundError
```
→ 기본 프롬프트가 자동으로 사용됨

### 3. GPU 메모리 부족
```
❌ CUDA out of memory
```
→ `--num_samples` 값을 줄이거나 `--efficiency`만 측정

### 4. 렌더링 오류
```
❌ Blender 렌더링 실패
```
→ Blender 설치 상태 확인 (자동 설치 시도됨)

## 📈 결과 해석

### CLIP Score
- **80+ 점**: 🟢 우수 (높은 텍스트-3D 일관성)
- **70-80 점**: 🟡 양호
- **60-70 점**: 🟠 보통  
- **< 60 점**: 🔴 개선 필요

### Fréchet Distance
- **≤ 20**: 🟢 우수 (높은 생성 품질)
- **20-40**: 🟡 양호
- **40-60**: 🟠 보통
- **> 60**: 🔴 개선 필요

### 효율성 지표
- **모델 크기**: 작을수록 좋음 (양자화 효과)
- **추론 시간**: 짧을수록 좋음 (속도 개선)
- **GPU 메모리**: 적을수록 좋음 (메모리 효율성)

## 🎯 권장 워크플로우

1. **효율성 체크**: 먼저 `--efficiency`로 빠르게 모델 사이즈/속도 확인
2. **품질 검증**: `--CLIP --num_samples 20`으로 빠른 품질 체크
3. **상세 평가**: 모든 지표로 최종 평가 `--CLIP --FD --efficiency --num_samples 100`

## 📝 배치 평가 스크립트 예시

```bash
#!/bin/bash
# evaluate_all_models.sh

MODELS=(
    "quantization_results/trellis_text-base_quantized"
    "quantization_results/trellis_text-large_quantized"  
    "quantization_results/custom_quantized_model"
)

for model in "${MODELS[@]}"; do
    echo "📋 평가 중: $model"
    python evaluate_quantized_model.py \
        --model_path "$model" \
        --dataset datasets/Toys4k \
        --CLIP --FD --efficiency \
        --num_samples 50 \
        --output_dir "evaluation_results/$(basename $model)"
    echo "✅ 완료: $model"
    echo ""
done
```

이 도구를 사용하면 양자화된 모델들을 체계적으로 평가하고 성능을 비교할 수 있습니다! 🚀