#!/usr/bin/env python3
"""
Simple Unicode test
"""

import re

def simple_unicode_test():
    test_line = '1.  "Pink-brown plane with small, delicate wings, resting on a mossy forest floor, sunlight dappling through leaves, highly detailed, realistic rendering."'
    
    # Just search for the Unicode characters
    left_quote = '\u201c'  # "
    right_quote = '\u201d' # "
    
    print(f"Test line length: {len(test_line)}")
    print(f"Looking for left quote {repr(left_quote)} (ord {ord(left_quote)})")
    print(f"Looking for right quote {repr(right_quote)} (ord {ord(right_quote)})")
    print()
    
    # Check if the characters exist
    if left_quote in test_line:
        pos = test_line.find(left_quote)
        print(f"Left quote found at position {pos}")
    else:
        print("Left quote NOT found")
    
    if right_quote in test_line:
        pos = test_line.find(right_quote)
        print(f"Right quote found at position {pos}")
    else:
        print("Right quote NOT found")
    
    # Try simple regex search
    left_search = re.search(left_quote, test_line)
    print(f"Left quote regex search: {left_search}")
    
    right_search = re.search(right_quote, test_line)
    print(f"Right quote regex search: {right_search}")

if __name__ == "__main__":
    simple_unicode_test()