#\!/usr/bin/env python3
import pandas as pd

df = pd.read_excel('prompt_generation_outputs/automated_prompt_results.xlsx')
print('=== Results Summary ===')
print(f'Total rows: {len(df)}')
print()

print('=== Object Names by Model ===')
for model in df['llm_model'].unique():
    model_data = df[df['llm_model'] == model]
    print(f'{model}:')
    for obj_name in model_data['object_name'].unique():
        count = len(model_data[model_data['object_name'] == obj_name])
        print(f'  {obj_name}: {count}')
    print()

print('=== SHA256 and file_identifier status ===')
for model in df['llm_model'].unique():
    model_data = df[df['llm_model'] == model]
    empty_sha256 = len(model_data[model_data['sha256'] == ''])
    empty_file_id = len(model_data[model_data['file_identifier'] == ''])
    print(f'{model}: {empty_sha256} empty sha256, {empty_file_id} empty file_identifier')

print()
print('=== Text Prompt Samples ===')
for model in df['llm_model'].unique():
    model_data = df[df['llm_model'] == model]
    sample = model_data.iloc[0]
    print(f'{model}:')
    print(f'  text_prompt: {sample["text_prompt"][:100]}...')
    print()

print('=== First few rows ===')
print(df[['llm_model', 'object_name', 'text_prompt', 'sha256', 'file_identifier']].head(10))
EOF < /dev/null
