import pandas as pd

df = pd.read_excel('prompt_generation_outputs/automated_prompt_results.xlsx')
deepseek_row = df[df['llm_model'].str.contains('deepseek', case=False, na=False)].iloc[0]
print('DeepSeek row details:')
for col in df.columns:
    print(f'{col}: {deepseek_row[col]}')