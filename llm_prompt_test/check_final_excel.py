import pandas as pd

df = pd.read_excel('prompt_generation_outputs/automated_prompt_results.xlsx')
print('Total rows:', len(df))
print('\nAll DeepSeek rows:')
deepseek_rows = df[df['llm_model'].str.contains('deepseek', case=False, na=False)]
for idx, row in deepseek_rows.iterrows():
    print(f'\nRow {idx + 1}:')
    print(f'  object_name: {row["object_name"]}')
    print(f'  user_prompt: {row["user_prompt"]}')  
    print(f'  text_prompt: {row["text_prompt"]}')
    print(f'  llm_model: {row["llm_model"]}')
    print(f'  file_identifier: {row["file_identifier"]}')