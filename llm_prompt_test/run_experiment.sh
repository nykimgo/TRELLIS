#!/bin/bash

# GPU 실험 스크립트
# 작은 모델: 3개씩, 중간 모델: 2개씩, 큰 모델: 1개씩 배치

# 작업 디렉토리 설정
cd /home/sr/TRELLIS/llm_prompt_test

# CSV 파일 목록
CSV_FILES=(
#   "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part01.csv"
#  "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part02.csv"
#    "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part03.csv"
#    "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part04.csv"
    "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part05.csv"
)

# 작은 모델 배치 (3개씩)
SMALL_BATCHES=(
    '["gemma3:1b", "qwen3:0.6b"]'
    '["llama3.1:8b", "deepseek-r1:1.5b"]'
)

# 중간 모델 배치 (2개씩)
MEDIUM_BATCHES=(
    '["qwen3:14b", "gemma3:12b"]'
    '["gpt-oss:20b", "deepseek-r1:14b-qwen-distill-q8_0"]'
)

# 큰 모델 배치 (1개씩)
LARGE_BATCHES=(
    '["gemma3:27b-it-q8_0"]'
    '["deepseek-r1:32b-qwen-distill-q8_0"]'
    '["qwen3:32b-q8_0"]'
    '["llama3.1:70b-instruct-q4_0"]'
)

# 설정 파일 업데이트 함수
update_config() {
    local models="$1"
    cat > pipeline_config.json << EOF
{
    "metadata_path": "../datasets/HSSD/metadata.csv",
    "num_samples": 100,
    "models": $models,
    "normalize_output": true,
    "generate_object_names": true,
    "comment": "Configuration for automated prompt augmentation pipeline"
}
EOF
}

echo "=== GPU 실험 시작 ==="

# 각 CSV 파일에 대해 실험 실행
for csv_file in "${CSV_FILES[@]}"; do
    echo "처리 중: $csv_file"
    
    # 작은 모델 배치 실행
    echo "--- 작은 모델 배치 실행 ---"
    for batch in "${SMALL_BATCHES[@]}"; do
        echo "모델 배치: $batch"
        update_config "$batch"
        python run_automated_pipeline.py --use-csv "$csv_file"
        echo "배치 완료"
    done
    
    # 중간 모델 배치 실행
    echo "--- 중간 모델 배치 실행 ---"
    for batch in "${MEDIUM_BATCHES[@]}"; do
        echo "모델 배치: $batch"
        update_config "$batch"
        python run_automated_pipeline.py --use-csv "$csv_file"
        echo "배치 완료"
    done
    
    # 큰 모델 배치 실행
    echo "--- 큰 모델 배치 실행 ---"
    for batch in "${LARGE_BATCHES[@]}"; do
        echo "모델 배치: $batch"
        update_config "$batch"
        python run_automated_pipeline.py --use-csv "$csv_file"
        echo "배치 완료"
    done
    
    echo "CSV 파일 처리 완료: $csv_file"
    echo ""
done

echo "=== 모든 실험 완료 ==="