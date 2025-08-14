#!/usr/bin/env python3
"""
Test the Unicode pattern separately
"""

import re

def test_unicode_pattern():
    test_line = '1.  "Pink-brown plane with small, delicate wings, resting on a mossy forest floor, sunlight dappling through leaves, highly detailed, realistic rendering."'
    
    print(f"Test line: {repr(test_line)}")
    
    pattern = r'^(\d+)\.\s*[""](.+?)[""]'
    
    match = re.match(pattern, test_line)
    
    if match:
        print(f"✓ Match found: {match.groups()}")
    else:
        print("✗ No match")
    
    # Try with findall
    matches = re.findall(pattern, test_line)
    print(f"Findall result: {matches}")

if __name__ == "__main__":
    test_unicode_pattern()