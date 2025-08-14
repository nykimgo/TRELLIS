# TRELLIS CLIP Score Evaluation

이 폴더는 TRELLIS 3D 생성 결과의 CLIP Score 평가를 위한 코드들을 포함합니다.

## 📁 **파일 구조**

### **메인 평가 스크립트**
- **`toys4k_clip_evaluator_fast.py`** - **권장** 빠른 Toys4k 평가기
- **`toys4k_clip_evaluator.py`** - 원본 Toys4k 평가기 (느림)
- **`clip_score_evaluator.py`** - 범용 CLIP 평가기

### **테스트 및 개발용**
- **`toys4k_clip_evaluator_no_clip.py`** - CLIP 없이 mock 점수 생성
- **`test_basic_functionality.py`** - 기본 기능 테스트
- **`test_toys4k_evaluator.py`** - Toys4k 평가기 테스트
- **`example_clip_evaluation.py`** - 사용 예시

### **문서**
- **`requirements_clip_eval.txt`** - 필요한 패키지 목록
- **`CLIP_SCORE_EVALUATION_README.md`** - 상세 문서
- **`FINAL_IMPLEMENTATION_SUMMARY.md`** - 구현 요약

## 🚀 **빠른 시작**

### **1. 의존성 설치**
```bash
cd evaluation
pip install -r requirements_clip_eval.txt
```

### **2. 테스트 실행 (5개 에셋)**
```bash
python toys4k_clip_evaluator_fast.py --max_assets 5 --output_path test_results.csv
```

### **3. 전체 평가 실행**
```bash
python toys4k_clip_evaluator_fast.py --output_path full_toys4k_clip_scores.csv
```

## 📊 **기대 결과**

```
=== Toys4k CLIP Score Evaluation Results ===
Total assets: 5
Successful evaluations: 5
Success rate: 100.00%
Mean CLIP Score: 0.1929
Mean CLIP Score (×100): 19.29
```

## ⚡ **성능 비교**

| 스크립트 | 렌더링 방식 | 속도/에셋 | 전체 시간 |
|----------|-------------|-----------|-----------|
| `toys4k_clip_evaluator_fast.py` | trimesh | ~1.5초 | ~1.5시간 |
| `toys4k_clip_evaluator.py` | matplotlib | ~40초 | ~36시간 |

## 🎯 **사용 권장사항**

### **실제 평가용**
```bash
python toys4k_clip_evaluator_fast.py --output_path results.csv
```

### **개발/테스트용**
```bash
python toys4k_clip_evaluator_no_clip.py --max_assets 10
```

### **기능 확인용**
```bash
python test_basic_functionality.py
```

## 📄 **출력 파일**

- **`results.csv`** - 상세 에셋별 결과
- **`results_summary.json`** - 집계된 메트릭스
- 최종 CLIP Score는 × 100 스케일로 보고됨

## 🔧 **문제 해결**

- **CLIP 모델 로딩 에러**: safetensors 지원 모델 자동 사용
- **렌더링 느림**: fast 버전 사용 권장
- **메모리 부족**: `--max_assets` 옵션으로 배치 처리

자세한 내용은 `CLIP_SCORE_EVALUATION_README.md`를 참조하세요.