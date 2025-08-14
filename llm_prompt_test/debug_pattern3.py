#!/usr/bin/env python3
"""
Debug pattern 3 specifically
"""

import re

def debug_pattern3():
    test_line = '1.  "Pink-brown plane with small, delicate wings, resting on a mossy forest floor, sunlight dappling through leaves, highly detailed, realistic rendering."'
    
    print(f"Test line: {repr(test_line)}")
    
    # Original pattern 3
    pattern3 = r'(\d+)\.\s*"([^"]+)"'
    match = re.match(pattern3, test_line)
    print(f"Pattern 3 match: {match}")
    
    if match:
        print(f"Groups: {match.groups()}")
    
    # Test with findall
    matches = re.findall(pattern3, test_line)
    print(f"Findall matches: {matches}")
    
    # Try a more flexible pattern
    pattern_flexible = r'(\d+)\.\s*"([^"]*)"'  # Allow empty content
    match_flex = re.match(pattern_flexible, test_line)
    print(f"Flexible pattern match: {match_flex}")
    
    if match_flex:
        print(f"Flexible groups: {match_flex.groups()}")

if __name__ == "__main__":
    debug_pattern3()