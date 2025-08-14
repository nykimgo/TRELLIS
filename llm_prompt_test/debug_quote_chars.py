#!/usr/bin/env python3
"""
Check the exact Unicode characters in the quotes
"""

def debug_quote_chars():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the first quote character
    lines = content.split('\n')
    for line in lines:
        if '1.' in line:
            print(f"Line: {repr(line)}")
            
            # Find quote characters
            for i, char in enumerate(line):
                if char not in '1234567890. abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,-':
                    print(f"Character at position {i}: {repr(char)} (ord: {ord(char)}) (unicode: \\u{ord(char):04x})")
            break
    
    # Also check what the Unicode quotes should be
    print("\nExpected Unicode quotes:")
    left_quote = '"'
    right_quote = '"' 
    ascii_quote = '"'
    print(f"Left double quote: {repr(left_quote)} (ord: {ord(left_quote)}) (unicode: \\u{ord(left_quote):04x})")
    print(f"Right double quote: {repr(right_quote)} (ord: {ord(right_quote)}) (unicode: \\u{ord(right_quote):04x})")
    print(f"ASCII quote: {repr(ascii_quote)} (ord: {ord(ascii_quote)}) (unicode: \\u{ord(ascii_quote):04x})")

if __name__ == "__main__":
    debug_quote_chars()