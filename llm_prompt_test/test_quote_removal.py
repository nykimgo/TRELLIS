#!/usr/bin/env python3
"""
Test the quote removal function
"""

import sys
import os
sys.path.append('/home/sr/TRELLIS/llm_prompt_test')

from automated_prompt_generator import PromptGenerator

def test_quote_removal():
    generator = PromptGenerator("/mnt/nas/Benchmark_Datatset/Toys4k/metadata.csv", 3)
    
    # Test cases
    test_cases = [
        '"Pink-brown plane with small, delicate wings"',  # ASCII quotes
        '"Pink-brown plane with small, delicate wings"',  # Unicode left+right quotes  
        '"Pink-brown plane with small, delicate wings"',  # Left quote only
        'Pink-brown plane with small, delicate wings',    # No quotes
        '"Mixed quote types"',                            # Mixed ASCII and Unicode
    ]
    
    print("Testing quote removal function:")
    print("=" * 50)
    
    for test_case in test_cases:
        result = generator._remove_quotes(test_case)
        
        # Check for remaining quotes
        has_start_quote = result.startswith(('"', '"', '"'))
        has_end_quote = result.endswith(('"', '"', '"'))
        
        print(f"Input:  {repr(test_case)}")
        print(f"Output: {repr(result)}")
        print(f"Status: {'❌ Still has quotes' if (has_start_quote or has_end_quote) else '✅ Clean'}")
        print()

if __name__ == "__main__":
    test_quote_removal()