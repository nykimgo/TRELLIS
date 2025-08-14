#!/usr/bin/env python3
"""
Test pattern with newline handling
"""

import re

def test_with_newline():
    test_content = '1.  "Pink-brown plane with small, delicate wings, resting on a mossy forest floor, sunlight dappling through leaves, highly detailed, realistic rendering."\n'
    
    patterns = [
        r'(\d+)\.\s*"([^"]+)"',       # Original - fails
        r'(\d+)\.\s*"([^"\n]+)"',     # Exclude newlines explicitly
        r'(\d+)\.\s*"([^"]*?)"',      # Non-greedy with any chars except quote
        r'(\d+)\.\s*"(.*?)"',         # Non-greedy any char
    ]
    
    print(f"Test content: {repr(test_content)}")
    
    for i, pattern in enumerate(patterns, 1):
        print(f"\nPattern {i}: {pattern}")
        matches = re.findall(pattern, test_content)
        print(f"  Matches: {matches}")
        
        search_result = re.search(pattern, test_content)
        if search_result:
            print(f"  Groups: {search_result.groups()}")

if __name__ == "__main__":
    test_with_newline()