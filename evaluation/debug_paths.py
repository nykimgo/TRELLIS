import pandas as pd
from pathlib import Path

# Load merged data
sampled_df = pd.read_csv('/mnt/nas/tmp/nayeon/sampled_data_100_random.csv')
results_df = pd.read_excel('/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx')
merged_df = pd.merge(sampled_df, results_df, on='sha256', how='inner')

# Check first few rows
for i in range(2):
    row = merged_df.iloc[i]
    file_identifier = row['file_identifier_x']
    llm_model = row['llm_model']
    object_name_clean = row['object_name_clean']
    
    # Convert LLM model name
    llm_model_folder = llm_model.replace(':', '_')
    
    print(f'Row {i}:')
    print(f'  file_identifier: {file_identifier}')
    print(f'  llm_model: {llm_model} -> {llm_model_folder}')
    print(f'  object_name_clean: {object_name_clean}')
    print(f'  Ground truth path: /mnt/nas/tmp/nayeon/toys4k/{file_identifier}')
    print(f'  Generated path: /mnt/nas/tmp/nayeon/trellis_results/{llm_model_folder}/{object_name_clean}/')
    print()