#!/usr/bin/env python3
"""
Test with the correct Unicode characters
"""

import re

def test_correct_unicode():
    test_line = '1.  "Pink-brown plane with small, delicate wings, resting on a mossy forest floor, sunlight dappling through leaves, highly detailed, realistic rendering."'
    
    print(f"Test line: {repr(test_line)}")
    
    # Use the actual Unicode characters
    left_quote = '\u201c'  # "
    right_quote = '\u201d' # "
    
    pattern = rf'^(\d+)\.\s*{left_quote}(.+?){right_quote}'
    print(f"Pattern: {pattern}")
    
    match = re.match(pattern, test_line)
    
    if match:
        print(f"✓ Match found: {match.groups()}")
    else:
        print("✗ No match")
        
        # Try individual quotes
        pattern_left = rf'^(\d+)\.\s*{left_quote}(.+)'
        match_left = re.match(pattern_left, test_line)
        if match_left:
            print(f"  Left quote works: {match_left.groups()[0]}, {match_left.groups()[1][:50]}...")
        
        pattern_right = rf'(.+){right_quote}$'
        match_right = re.search(pattern_right, test_line)
        if match_right:
            print(f"  Right quote works: ...{match_right.groups()[0][-50:]}")

if __name__ == "__main__":
    test_correct_unicode()