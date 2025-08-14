# 자동 프롬프트 증강 시스템 사용 가이드

이 시스템은 Toys4k 데이터셋의 텍스트 프롬프트를 LLM을 사용하여 자동으로 증강하는 파이프라인입니다.

## 📋 시스템 구성

### 주요 파일들
- `automated_prompt_generator.py`: 메인 프롬프트 생성 엔진
- `llm_output_normalizer.py`: LLM 출력 정규화 도구
- `object_name_generator.py`: 객체명 자동 생성 도구
- `run_automated_pipeline.py`: 통합 파이프라인 실행기
- `test_pipeline.py`: 시스템 테스트 스크립트
- `pipeline_config.json`: 설정 파일

## 🚀 빠른 시작

### 1. 의존성 설치
```bash
# Python 패키지 설치
pip install pandas openpyxl

# Ollama 설치 (https://ollama.ai)
# Linux/macOS:
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: ollama.ai에서 다운로드
```

### 2. Ollama 모델 설치
```bash
# 권장 모델들 설치
ollama pull gemma3:1b          # 빠른 처리용
ollama pull qwen3:1.7b         # 균형잡힌 성능
ollama pull qwen3:14b          # 고품질 출력용
ollama pull deepseek-r1:1.5b   # 추론 특화
```

### 3. 시스템 테스트
```bash
python test_pipeline.py
```

### 4. 외부 API 설정 (초고속 처리)
```bash
# 🚀 ULTRA FAST: 외부 API 설정 (강력 추천!)
python setup_api.py

# Groq (무료, 초고속) 또는 OpenAI (유료, 안정적) 설정
# 5-50배 빠른 속도!
```

### 5. 파이프라인 실행
```bash
# 인터랙티브 모드로 실행 (데이터셋 선택)
python run_automated_pipeline.py

# 특정 메타데이터 파일로 실행
python run_automated_pipeline.py ../datasets/HSSD/metadata.csv 50

# 테스트 실행 (데이터셋 선택)
python test_pipeline.py
```

## ⚡ 외부 API 설정 (초고속 처리)

로컬 Ollama 모델 대신 외부 API를 사용하면 **5-50배 빠른 속도**를 경험할 수 있습니다!

### 🚀 추천 API 순서 (속도 + 비용 기준)

| API | 속도 | 비용 | 특징 |
|-----|------|------|------|
| **Groq** ⭐ | 초고속 | 무료 티어 | Llama 3.1 70B, 무료로 빠른 처리 |
| **OpenAI** | 빠름 | 유료 | GPT-4o-mini, 가장 안정적 |
| **Claude** | 빠름 | 유료 | 높은 품질, 좋은 추론 능력 |
| **Gemini** | 보통 | 무료/유료 | Google 모델, 무료 티어 |

### 🔧 API 설정 방법

#### 방법 1: 자동 설정 (추천)
```bash
python setup_api.py
```

#### 방법 2: 수동 설정
`api_config.json` 파일을 직접 편집:

```json
{
  "preferred_apis": ["groq", "openai"],
  "groq": {
    "api_key": "your_groq_api_key_here",
    "enabled": true
  },
  "openai": {
    "api_key": "your_openai_api_key_here", 
    "enabled": true
  }
}
```

### 🔑 API 키 발급 방법

- **Groq**: https://console.groq.com (무료 계정으로 충분)
- **OpenAI**: https://platform.openai.com (신용카드 필요)
- **Claude**: https://console.anthropic.com (신용카드 필요)
- **Gemini**: https://makersuite.google.com (무료 티어 있음)

### ⚡ 성능 비교

| 처리 방식 | 100개 샘플 소요 시간 | 비용 |
|-----------|-------------------|------|
| 로컬 Ollama | 10-30분 | 무료 (GPU 필요) |
| **Groq API** | **1-3분** | **무료** |
| OpenAI API | 2-5분 | ~$0.50 |
| Claude API | 2-5분 | ~$0.30 |

## ⚙️ 설정 사용자화

`pipeline_config.json` 파일을 수정하여 설정을 변경할 수 있습니다:

```json
{
    "metadata_path": "datasets/HSSD/metadata.csv",
    "num_samples": 100,
    "models": [
        "gemma3:1b",
        "qwen3:1.7b",
        "qwen3:14b"
    ],
    "normalize_output": true,
    "generate_object_names": true
}
```

## 📊 출력 결과

파이프라인 실행 후 `prompt_generation_outputs/` 디렉토리에 다음 파일들이 생성됩니다:

### 중간 결과물
- `prompts.txt`: LLM에 입력할 프롬프트 파일
- `detailed_prompt_[모델명].txt`: 각 모델별 원시 출력
- `detailed_prompt_[모델명]_normalized.txt`: 정규화된 출력

### 최종 결과물
- `automated_prompt_results.xlsx`: 기본 결과 Excel 파일
- `automated_prompt_results_with_object_names.xlsx`: 객체명이 추가된 최종 Excel 파일

### Excel 파일 구조
| 컬럼명 | 설명 |
|--------|------|
| category | LLM 모델 카테고리 |
| llm_model | 사용된 LLM 모델명 |
| object_name | 추출된 주요 객체명 |
| user_prompt | 원본 짧은 프롬프트 |
| text_prompt | LLM이 증강한 상세 프롬프트 |
| sha256 | 원본 데이터의 해시값 |
| file_identifier | 파일 식별자 |

## 🔧 개별 도구 사용법

### 1. 프롬프트 생성만 실행
```python
from automated_prompt_generator import PromptGenerator

generator = PromptGenerator("datasets/HSSD/metadata.csv", num_samples=50)
excel_path = generator.run_full_pipeline(["gemma3:1b", "qwen3:1.7b"])
```

### 2. LLM 출력 정규화
```bash
python llm_output_normalizer.py prompt_generation_outputs/
```

### 3. 객체명 생성
```bash
# Excel 파일에 객체명 추가
python object_name_generator.py results.xlsx

# 단일 프롬프트 테스트
python object_name_generator.py "dark wooden table with drawer"
```

## 🎯 기능 상세

### 자동 캡션 선택
- metadata.csv에서 4-8단어 길이의 짧은 캡션 자동 선택
- 가장 간결한 캡션을 우선적으로 선택
- 랜덤 샘플링으로 다양성 확보

### LLM 프롬프트 증강
- 원본 짧은 캡션을 40단어 이내의 상세한 3D 생성용 프롬프트로 변환
- 색상, 형태, 스타일, 재질, 환경 등의 시각적 세부사항 추가
- 3D 모델 생성에 최적화된 명확하고 구체적인 설명

### 출력 정규화
- 다양한 LLM 출력 형식을 통일된 `"원본":"증강"` 형태로 변환
- LLM 기반 지능형 정규화와 규칙 기반 fallback 제공
- 번호, 특수문자, 불필요한 포맷팅 자동 제거

### 객체명 추출
- LLM을 사용한 지능형 주요 객체 식별
- 가구, 조명, 가전, 장식품 등 카테고리별 키워드 데이터베이스
- Rule-based fallback으로 안정성 확보

## 🔍 문제 해결

### 일반적인 문제들

#### 1. Ollama 모델 설치 오류
```bash
# 모델 목록 확인
ollama list

# 특정 모델 재설치
ollama pull gemma3:1b
```

#### 2. 메모리 부족 오류
- 더 작은 모델 사용 (gemma3:1b)
- 샘플 수 줄이기 (num_samples)
- 모델을 하나씩 순차 실행

#### 3. LLM 출력 파싱 오류
- 출력 정규화 활성화: `normalize_output: true`
- 수동 정규화 실행: `python llm_output_normalizer.py output_dir/`

#### 4. 객체명 생성 실패
- LLM 모델 확인 및 재설치
- Fallback 모드 사용 (자동으로 적용됨)

### 로그 확인
각 단계별 상세 로그가 콘솔에 출력됩니다. 오류 발생시 로그를 확인하여 문제점을 파악하세요.

## 📈 성능 최적화

### 모델 선택 가이드
- **빠른 처리**: `gemma3:1b`, `qwen3:0.6b`
- **균형잡힌 성능**: `qwen3:1.7b`, `deepseek-r1:1.5b`
- **고품질 출력**: `qwen3:14b`, `qwen3:32b`

### 배치 크기 조정
- GPU 메모리에 따라 샘플 수 조정
- 대용량 처리시 여러 번으로 나누어 실행

### 병렬 처리
- 여러 모델을 동시에 실행하여 시간 단축
- CPU 코어 수에 맞춰 동시 실행 모델 수 조정

## 🔗 확장 가능성

### 새로운 LLM 모델 추가
1. Ollama에 모델 설치: `ollama pull [model_name]`
2. `pipeline_config.json`의 `models` 배열에 추가
3. 파이프라인 재실행

### 커스텀 프롬프트 템플릿
`automated_prompt_generator.py`의 `prompt_template` 수정하여 다른 스타일의 증강 가능

### 다른 데이터셋 적용
- CSV 형식의 캡션 데이터가 있는 모든 데이터셋에 적용 가능
- 컬럼명만 맞춰주면 자동 처리

## 📞 지원

문제가 발생하거나 개선 사항이 있다면:
1. 먼저 `test_pipeline.py`로 시스템 상태 확인
2. 로그 메시지를 통해 오류 위치 파악
3. 설정 파일과 입력 데이터 형식 재확인

---

**참고**: 이 시스템은 기존 수동 프로세스를 자동화하여 효율성을 크게 향상시킵니다. 100개 샘플 기준으로 수동 작업시 몇 시간이 걸리던 작업을 10-30분 내에 완료할 수 있습니다.