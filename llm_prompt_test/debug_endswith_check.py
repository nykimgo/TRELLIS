#!/usr/bin/env python3
"""
Debug the endswith check specifically
"""

import re

def debug_endswith_check():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        if re.match(r'^1\.', line):  # Only check first line
            print(f"Full line: {repr(line)}")
            
            match = re.match(r'^(\d+)\.\s*(?:\d+\.\s*)?(.+)$', line)
            if match:
                num_str, rest = match.groups()
                print(f"num_str: {repr(num_str)}")
                print(f"rest: {repr(rest)}")
                print(f"rest length: {len(rest)}")
                
                quote_chars = ('"', '"', '"')
                
                # Check start
                starts_with = rest.startswith(quote_chars)
                print(f"starts_with_quote: {starts_with}")
                if starts_with:
                    for i, char in enumerate(quote_chars):
                        if rest.startswith(char):
                            print(f"  Starts with {repr(char)} (ord: {ord(char)})")
                
                # Check end
                ends_with = rest.endswith(quote_chars)
                print(f"ends_with_quote: {ends_with}")
                if not ends_with:
                    print(f"Last few characters: {repr(rest[-10:])}")
                    for i, char in enumerate(rest[-5:]):
                        print(f"  rest[-{5-i}]: {repr(char)} (ord: {ord(char)})")
                
                # Show the actual end characters to see what's there
                if rest:
                    last_char = rest[-1]
                    print(f"Last character: {repr(last_char)} (ord: {ord(last_char)})")
                    
                    # Check if it's any of our quote chars
                    for char in quote_chars:
                        if last_char == char:
                            print(f"  ✓ Last char matches {repr(char)}")
                        else:
                            print(f"  ✗ Last char doesn't match {repr(char)}")
            break

if __name__ == "__main__":
    debug_endswith_check()