#!/bin/bash
# TRELLIS QLoRA 분산 훈련 스크립트 (4 GPUs)

set -e

# 환경 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/.."
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 분산 훈련 설정
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=4

# 기본 파라미터
MODEL="text-large"
CONFIG="configs/qlora_config.yaml"
OUTPUT_DIR="./qlora_experiments"

echo "🚀 TRELLIS QLoRA 분산 훈련 (4 GPUs)"
echo "=" * 50

# GPU 상태 확인
echo "🔍 GPU 상태:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits

echo ""
echo "📋 분산 설정:"
echo "  MASTER_ADDR: $MASTER_ADDR"
echo "  MASTER_PORT: $MASTER_PORT"
echo "  WORLD_SIZE: $WORLD_SIZE"
echo "  모델: $MODEL"
echo "=" * 50

# 각 GPU에서 프로세스 시작
for i in {0..3}; do
    export RANK=$i
    echo "🔄 GPU $i 에서 프로세스 시작 (Rank $RANK)..."
    
    python main.py \
        --model "$MODEL" \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR" \
        --num_gpus 4 \
        --fp16 \
        --gradient_checkpointing &
    
    # 프로세스 간 간격
    sleep 2
done

echo "⏳ 모든 프로세스 시작됨. 완료까지 대기 중..."

# 모든 백그라운드 프로세스 완료 대기
wait

echo "✅ 분산 훈련 완료!"

# 결과 확인
if [ -d "$OUTPUT_DIR" ]; then
    echo "📊 결과 디렉토리: $OUTPUT_DIR"
    find "$OUTPUT_DIR" -name "*.log" -o -name "*.json" -o -name "*.pt" | head -10
fi