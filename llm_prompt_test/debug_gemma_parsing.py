#!/usr/bin/env python3
"""
Debug script to understand why gemma parsing is failing
"""

import re

def debug_parsing():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Content:")
    print(repr(content))
    print("\n" + "="*50 + "\n")
    
    # Test pattern 1
    print("Testing pattern 1: '\"번호. 원본\":\"증강\"'")
    pattern1 = r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"'
    matches1 = re.findall(pattern1, content, re.MULTILINE | re.DOTALL)
    print(f"Pattern 1 matches: {matches1}")
    print()
    
    # Test pattern 2
    print("Testing pattern 2: '번호. \"원본\":\"증강\"'")
    pattern2 = r'\d+\.\s*"([^"]+)"\s*:\s*"([^"]+)"'
    matches2 = re.findall(pattern2, content, re.MULTILINE | re.DOTALL)
    print(f"Pattern 2 matches: {matches2}")
    print()
    
    # Test pattern 3 (qwen format)
    print("Testing pattern 3: qwen format")
    pattern3 = r'(\d+)\.\s*"([^"]+)"'
    matches3 = re.findall(pattern3, content, re.MULTILINE | re.DOTALL)
    print(f"Pattern 3 matches: {matches3}")
    print()
    
    # Test pattern 4 (gemma format - line by line)
    print("Testing pattern 4: gemma format line by line")
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        print(f"Line {i}: {repr(line)}")
        
        match = re.match(r'^(\d+)\.\s*"(.+)"', line)
        if match:
            num_str, enhanced = match.groups()
            print(f"  -> Matched: num={num_str}, enhanced={enhanced}")
    
if __name__ == "__main__":
    debug_parsing()