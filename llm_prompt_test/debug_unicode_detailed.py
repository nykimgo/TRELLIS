#!/usr/bin/env python3
"""
Debug Unicode pattern in detail
"""

import re

def debug_unicode_detailed():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the first numbered line
    for line in lines:
        line = line.strip()
        if line.startswith('1.'):
            print(f"Found line: {repr(line)}")
            
            # Check the quote characters
            for i, char in enumerate(line):
                print(f"  {i:2d}: {repr(char):>6} (ord: {ord(char):>5}) (unicode: \\u{ord(char):04x})")
            
            # Test the pattern
            pattern = r'^(\d+)\.\s*[""](.+?)[""]'
            print(f"\nTesting pattern: {pattern}")
            
            match = re.match(pattern, line)
            if match:
                print(f"✓ Match: {match.groups()}")
            else:
                print("✗ No match")
                
                # Try variations
                variations = [
                    r'^(\d+)\.\s*"(.+?)"',  # Only left Unicode quote
                    r'^(\d+)\.\s*"(.+?)"',  # Only right Unicode quote  
                    r'^(\d+)\.\s*"(.+)"',   # Left Unicode quote, any end
                    r'^(\d+)\.\s*(.+?)$',   # Capture everything to end
                ]
                
                for j, var_pattern in enumerate(variations, 1):
                    var_match = re.match(var_pattern, line)
                    if var_match:
                        print(f"  Variation {j} ({var_pattern}): ✓ {var_match.groups()}")
                    else:
                        print(f"  Variation {j} ({var_pattern}): ✗")
            break

if __name__ == "__main__":
    debug_unicode_detailed()