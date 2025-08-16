#!/usr/bin/env python3

import json
from enhanced_normalizer import EnhancedNormalizer
import pandas as pd

# JSON 데이터 로드
with open('/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/parsed_json_from_groq.json') as f:
    groq_data = json.load(f)

normalized_data = groq_data['normalized_data']
print(f'Loaded {len(normalized_data)} entries from Groq JSON')

# CSV 데이터 로드
csv_df = pd.read_csv('/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part01.csv')
metadata_dict = {row['original_caption']: row.to_dict() for _, row in csv_df.iterrows()}
print(f'Loaded {len(metadata_dict)} metadata entries')

print('First 3 normalized entries:')
for i, entry in enumerate(normalized_data[:3]):
    print(f'{i+1}. {entry["object_name"]}: {entry["text_prompt"][:50]}...')

print('\nFirst 3 metadata keys:')
for i, key in enumerate(list(metadata_dict.keys())[:3]):
    print(f'{i+1}. {key}')

# normalizer로 Excel 생성 테스트
normalizer = EnhancedNormalizer()
output_path = '/tmp/test_output.xlsx'
result_path = normalizer.create_unified_excel(normalized_data, metadata_dict, output_path)

# 결과 확인
result_df = pd.read_excel(result_path)
print(f'\nExcel contains {len(result_df)} rows')
print('First 3 rows:')
print(result_df[['object_name', 'text_prompt']].head(3))