#!/usr/bin/env python3
import pandas as pd

# 현재 엑셀 결과 확인
df = pd.read_excel("prompt_generation_outputs/automated_prompt_results.xlsx")

print("=== 전체 데이터 확인 ===")
print(f"총 행 수: {len(df)}")
print("\n=== 모델별 데이터 ===")
for model in df['llm_model'].unique():
    model_data = df[df['llm_model'] == model]
    print(f"\n{model} ({len(model_data)} rows):")
    for i, row in model_data.iterrows():
        print(f"  Row {i+1}:")
        print(f"    user_prompt: {row['user_prompt']}")
        print(f"    text_prompt: {row['text_prompt'][:100]}...")
        print(f"    object_name: {row['object_name']}")
        print(f"    sha256: {'YES' if row['sha256'] else 'NO'}")

print("\n=== qwen3 상세 확인 ===")
qwen3_data = df[df['llm_model'] == 'qwen3:0.6b']
if len(qwen3_data) == 0:
    print("qwen3 데이터가 없습니다!")
else:
    print(f"qwen3 데이터 개수: {len(qwen3_data)}")
    for i, row in qwen3_data.iterrows():
        print(f"Entry {i+1}:")
        print(f"  user_prompt: '{row['user_prompt']}'")
        print(f"  text_prompt: '{row['text_prompt']}'")
        print(f"  object_name: '{row['object_name']}'")
        print(f"  sha256: '{row['sha256']}'")
        print(f"  file_identifier: '{row['file_identifier']}'")
        print()