#!/usr/bin/env python3
"""
Check Unicode characters
"""

import re

def check_unicode_chars():
    test_line = '1.  "Pink-brown plane with small, delicate wings, resting on a mossy forest floor, sunlight dappling through leaves, highly detailed, realistic rendering."'
    
    left_quote = '\u201c'  # "
    right_quote = '\u201d' # "
    
    print(f"Left quote: {repr(left_quote)} (ord: {ord(left_quote)})")
    print(f"Right quote: {repr(right_quote)} (ord: {ord(right_quote)})")
    print()
    
    # Find the actual quotes in the test line
    for i, char in enumerate(test_line):
        if ord(char) == 8220:
            print(f"Found left quote at position {i}: {repr(char)}")
        elif ord(char) == 8221:
            print(f"Found right quote at position {i}: {repr(char)}")
    
    # Test if we can match just the left quote
    pattern_left = f'^(\d+)\.\s*{left_quote}'
    match_left = re.match(pattern_left, test_line)
    print(f"\nLeft quote pattern: {repr(pattern_left)}")
    print(f"Left quote match: {match_left}")
    
    # Test if we can find the right quote
    pattern_right = f'{right_quote}$'
    match_right = re.search(pattern_right, test_line)
    print(f"Right quote pattern: {repr(pattern_right)}")
    print(f"Right quote match: {match_right}")
    
    # Test full pattern
    full_pattern = f'^(\d+)\.\s*{left_quote}(.+?){right_quote}'
    match_full = re.match(full_pattern, test_line)
    print(f"Full pattern: {repr(full_pattern)}")
    print(f"Full match: {match_full}")

if __name__ == "__main__":
    check_unicode_chars()