#!/usr/bin/env python3
"""
Debug the current parsing with the actual gemma3 format
"""

import sys
import os
import re
sys.path.append('/home/sr/TRELLIS/llm_prompt_test')

from automated_prompt_generator import PromptGenerator

def debug_current_parsing():
    generator = PromptGenerator("/mnt/nas/Benchmark_Datatset/Toys4k/metadata.csv", 3)
    generator.selected_data = [
        {'original_caption': 'Pink-brown plane with small wings.', 'sha256': 'test1', 'file_identifier': 'test1.obj'},
        {'original_caption': 'Fan with wooden blades.', 'sha256': 'test2', 'file_identifier': 'test2.obj'},
        {'original_caption': 'Frosted bottle with golden cross.', 'sha256': 'test3', 'file_identifier': 'test3.obj'}
    ]
    
    # Test current gemma3 format
    test_line = '1.  "Pink-brown plane with small, delicate wings, crafted from polished silver and accented with tiny blue gems, soaring gracefully through a vibrant, cloud-filled sky."'
    
    print(f"Testing line: {repr(test_line)}")
    print()
    
    # Pattern 4 logic
    match = re.match(r'^(\d+)\.\s*(?:\d+\.\s*)?(.+)$', test_line)
    if match:
        num_str, rest = match.groups()
        print(f"Pattern 4 match:")
        print(f"  num_str: {repr(num_str)}")
        print(f"  rest: {repr(rest)}")
        
        quote_chars = ('"', '"', '"')
        starts_with_quote = rest.startswith(quote_chars)
        ends_with_quote = rest.endswith(quote_chars)
        
        print(f"  starts_with_quote: {starts_with_quote}")
        print(f"  ends_with_quote: {ends_with_quote}")
        print(f"  quote condition: {starts_with_quote and ends_with_quote}")
        
        if starts_with_quote and ends_with_quote:
            enhanced = rest[1:-1]  # Remove first/last char
            print(f"  after quote removal: {repr(enhanced)}")
            enhanced_clean = generator._remove_quotes(enhanced)
            print(f"  after _remove_quotes: {repr(enhanced_clean)}")
        else:
            print(f"  Taking else branch...")
            enhanced = rest.strip()
            print(f"  enhanced before _remove_quotes: {repr(enhanced)}")
            enhanced_clean = generator._remove_quotes(enhanced)
            print(f"  after _remove_quotes: {repr(enhanced_clean)}")

if __name__ == "__main__":
    debug_current_parsing()