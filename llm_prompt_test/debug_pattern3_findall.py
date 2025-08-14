#!/usr/bin/env python3
"""
Debug why pattern 3 findall is failing
"""

import re

def debug_pattern3_findall():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Content:")
    print(repr(content))
    print()
    
    # Pattern 3 from our code
    pattern3 = r'(\d+)\.\s*"([^"]+)"'
    print(f"Pattern: {pattern3}")
    
    matches3 = re.findall(pattern3, content, re.MULTILINE | re.DOTALL)
    print(f"Findall result: {matches3}")
    
    # Let's try without DOTALL to see if that's the issue
    matches3_no_dotall = re.findall(pattern3, content, re.MULTILINE)
    print(f"Findall without DOTALL: {matches3_no_dotall}")
    
    # Let's test the pattern on individual lines
    lines = content.split('\n')
    print("\nTesting on individual lines:")
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith(('1.', '2.', '3.')):
            print(f"Line {i}: {repr(line)}")
            match = re.search(pattern3, line)
            print(f"  Search result: {match}")
            if match:
                print(f"  Groups: {match.groups()}")
            
            findall_result = re.findall(pattern3, line)
            print(f"  Findall result: {findall_result}")

if __name__ == "__main__":
    debug_pattern3_findall()