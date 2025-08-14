#!/usr/bin/env python3
"""
Test different pattern variations
"""

import re

def test_patterns():
    test_line = '1.  "Pink-brown plane with small, delicate wings, resting on a mossy forest floor, sunlight dappling through leaves, highly detailed, realistic rendering."'
    
    print(f"Test line: {repr(test_line)}")
    print()
    
    patterns = [
        r'(\d+)\.\s*"([^"]+)"',  # Original
        r'(\d+)\.\s*"(.+?)"',    # Non-greedy
        r'(\d+)\.\s*"(.*?)"',    # Non-greedy with any char
        r'(\d+)\.\s*"([^"]*)"',  # Allow empty
        r'(\d+)\.\s+"(.+?)"',    # At least one space
        r'(\d+)\.\s*"(.+)"',     # Greedy any char
    ]
    
    for i, pattern in enumerate(patterns, 1):
        print(f"Pattern {i}: {pattern}")
        match = re.search(pattern, test_line)
        if match:
            print(f"  ✓ Match: {match.groups()}")
        else:
            print(f"  ✗ No match")
        print()

if __name__ == "__main__":
    test_patterns()