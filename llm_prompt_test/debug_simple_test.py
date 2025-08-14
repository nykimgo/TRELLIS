#!/usr/bin/env python3
"""
Simple test to understand the issue
"""

import re

def simple_test():
    # Create a simple test case
    test_content = '1.  "Pink-brown plane with small, delicate wings, resting on a mossy forest floor, sunlight dappling through leaves, highly detailed, realistic rendering."'
    
    pattern = r'(\d+)\.\s*"([^"]+)"'
    
    print(f"Test content: {repr(test_content)}")
    print(f"Pattern: {pattern}")
    
    # Test with findall
    matches = re.findall(pattern, test_content)
    print(f"Findall matches: {matches}")
    
    # Test with search
    search_result = re.search(pattern, test_content)
    print(f"Search result: {search_result}")
    if search_result:
        print(f"Search groups: {search_result.groups()}")
    
    # Now test with the actual line from the file
    print("\n=== Now with actual file line ===")
    
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the line that starts with "1."
    for i, line in enumerate(lines):
        if line.strip().startswith('1.'):
            print(f"Line {i}: {repr(line)}")
            print(f"Line stripped: {repr(line.strip())}")
            
            matches_line = re.findall(pattern, line)
            print(f"Findall on this line: {matches_line}")
            
            search_line = re.search(pattern, line)
            print(f"Search on this line: {search_line}")
            if search_line:
                print(f"Search groups: {search_line.groups()}")
            break

if __name__ == "__main__":
    simple_test()