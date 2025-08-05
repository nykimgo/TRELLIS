#!/bin/bash
# TRELLIS QLoRA 훈련 스크립트

set -e

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/.."
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 기본 설정
MODEL="text-large"
CONFIG="configs/qlora_config.yaml"
OUTPUT_DIR="./qlora_experiments"
EXPERIMENT_NAME=""

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --quick)
            # 빠른 테스트용 설정
            EXPERIMENT_NAME="quick_test"
            echo "🚀 빠른 테스트 모드"
            shift
            ;;
        --help)
            echo "사용법: $0 [옵션]"
            echo "옵션:"
            echo "  --model MODEL          모델 선택 (text-base, text-large, text-xlarge)"
            echo "  --config CONFIG        설정 파일 경로"
            echo "  --output_dir DIR       출력 디렉토리"
            echo "  --experiment_name NAME 실험 이름"
            echo "  --quick               빠른 테스트 모드"
            echo "  --help                도움말"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            exit 1
            ;;
    esac
done

echo "🔧 TRELLIS QLoRA 훈련 시작"
echo "=" * 40
echo "📋 설정:"
echo "  모델: $MODEL"
echo "  설정: $CONFIG"
echo "  출력: $OUTPUT_DIR"
echo "  실험: $EXPERIMENT_NAME"
echo "=" * 40

# GPU 확인
echo "🔍 GPU 상태:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# Python 스크립트 실행
python main.py \
    --model "$MODEL" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR" \
    ${EXPERIMENT_NAME:+--experiment_name "$EXPERIMENT_NAME"} \
    --num_gpus 4 \
    --fp16 \
    --gradient_checkpointing

echo "✅ 훈련 완료!"