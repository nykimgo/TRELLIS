#!/usr/bin/env python3
"""
Check the exact bytes of the file
"""

import re

def check_bytes():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'rb') as f:
        raw_content = f.read()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text_content = f.read()
    
    print("Raw bytes (first 500):")
    print(raw_content[:500])
    print()
    
    print("Text content (first 500):")
    print(repr(text_content[:500]))
    print()
    
    # Find the line with "1."
    lines = text_content.split('\n')
    for i, line in enumerate(lines):
        if '1.' in line and '"' in line:
            print(f"Found line {i}: {repr(line)}")
            print(f"Line bytes: {line.encode('utf-8')}")
            
            # Test the pattern on this exact line
            pattern = r'(\d+)\.\s*"([^"]+)"'
            matches = re.findall(pattern, line)
            print(f"Pattern matches: {matches}")
            break

if __name__ == "__main__":
    check_bytes()