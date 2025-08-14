#!/usr/bin/env python3
"""
Debug pattern 1 specifically
"""

import re

def debug_pattern1():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Test pattern 1
    print("Testing pattern 1: '\"번호. 원본\":\"증강\"'")
    pattern1 = r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"'
    matches1 = re.findall(pattern1, content, re.MULTILINE | re.DOTALL)
    print(f"Pattern 1 matches: {matches1}")
    print(f"Number of matches: {len(matches1)}")
    
    # Also test if there are any matches at all
    if re.search(pattern1, content, re.MULTILINE | re.DOTALL):
        print("Pattern 1 found at least one match with search()")
    else:
        print("Pattern 1 found no matches with search()")

if __name__ == "__main__":
    debug_pattern1()