#!/usr/bin/env python3
"""
Check if quote removal is consistent across all text_prompt results
"""

import pandas as pd

def check_quote_consistency():
    excel_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/automated_prompt_results.xlsx"
    
    # Excel 파일 읽기
    df = pd.read_excel(excel_path)
    
    print("Checking quote consistency in text_prompt column:")
    print("=" * 60)
    
    for i, row in df.iterrows():
        text_prompt = row['text_prompt']
        model = row['llm_model']
        user_prompt = row['user_prompt']
        
        # 쌍따옴표로 시작하거나 끝나는지 확인
        starts_with_quote = text_prompt.startswith(('"', '"', '"'))
        ends_with_quote = text_prompt.endswith(('"', '"', '"'))
        
        quote_status = ""
        if starts_with_quote and ends_with_quote:
            quote_status = "📝 HAS quotes"
        elif starts_with_quote or ends_with_quote:
            quote_status = "⚠️  PARTIAL quotes"
        else:
            quote_status = "✅ NO quotes"
        
        print(f"{model:12s} | {quote_status} | {user_prompt[:30]:<30s}")
        print(f"             | {text_prompt[:80]}...")
        print()

if __name__ == "__main__":
    check_quote_consistency()