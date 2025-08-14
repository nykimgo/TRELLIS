#!/usr/bin/env python3
"""
Debug Unicode quote pattern specifically
"""

import re

def debug_unicode_quotes():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the lines with numbers
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        if re.match(r'^\d+\.', line):
            print(f"Line {i}: {repr(line)}")
            
            # Test different Unicode quote patterns
            patterns = [
                r'^(\d+)\.\s*["""](.+?)["""]',  # Unicode quotes
                r'^(\d+)\.\s*[""](.+?)[""]',    # Just left/right quotes
                r'^(\d+)\.\s*"(.+?)"',          # ASCII quotes
                r'^(\d+)\.\s*["""](.+?)$',      # Start with Unicode quote, end of line
                r'^(\d+)\.\s*(.+)$',            # Any content after number
            ]
            
            for j, pattern in enumerate(patterns, 1):
                match = re.match(pattern, line)
                if match:
                    print(f"  Pattern {j} ({pattern}): ✓ {match.groups()}")
                else:
                    print(f"  Pattern {j} ({pattern}): ✗")
            print()

if __name__ == "__main__":
    debug_unicode_quotes()