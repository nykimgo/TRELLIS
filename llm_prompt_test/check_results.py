#!/usr/bin/env python3
"""
Check the results of the improved pipeline
"""

import pandas as pd

def check_results():
    excel_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/automated_prompt_results.xlsx"
    
    # Excel 파일 읽기
    df = pd.read_excel(excel_path)
    
    print("Results summary:")
    print(f"Total rows: {len(df)}")
    print(f"Unique models: {df['llm_model'].unique()}")
    print(f"Unique original prompts: {df['user_prompt'].nunique()}")
    print()
    
    print("Object names by model:")
    for model in df['llm_model'].unique():
        model_df = df[df['llm_model'] == model]
        print(f"\n{model}:")
        for i, row in model_df.iterrows():
            print(f"  {row['user_prompt']} -> {row['object_name']}")
    
    print("\nObject name comparison:")
    for prompt in df['user_prompt'].unique():
        prompt_df = df[df['user_prompt'] == prompt]
        object_names = prompt_df['object_name'].unique()
        print(f"'{prompt}':")
        for model in prompt_df['llm_model'].unique():
            model_row = prompt_df[prompt_df['llm_model'] == model].iloc[0]
            print(f"  {model}: {model_row['object_name']}")
        print()

if __name__ == "__main__":
    check_results()