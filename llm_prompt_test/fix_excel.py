#!/usr/bin/env python3

import json
from enhanced_normalizer import EnhancedNormalizer
import pandas as pd

# 이미 생성된 JSON 데이터 로드
with open('/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/parsed_json_from_groq.json') as f:
    groq_data = json.load(f)

normalized_data = groq_data['normalized_data']
print(f'Loaded {len(normalized_data)} entries from existing Groq JSON')

# CSV 메타데이터 로드
csv_df = pd.read_csv('/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part01.csv')
metadata_dict = {row['original_caption']: row.to_dict() for _, row in csv_df.iterrows()}
print(f'Loaded {len(metadata_dict)} metadata entries')

# Excel 파일 생성
normalizer = EnhancedNormalizer()
output_path = '/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part01/prompt_results_1_medium_models.xlsx'
result_path = normalizer.create_unified_excel(normalized_data, metadata_dict, output_path)

print(f'Excel file created at: {result_path}')

# 결과 확인
result_df = pd.read_excel(result_path)
print(f'Excel contains {len(result_df)} rows')
print('Sample data:')
print(result_df[['object_name', 'text_prompt']].head(3))