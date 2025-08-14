#!/usr/bin/env python3
"""
Test script to verify the improved parsing functionality
"""

import sys
import os
import re
sys.path.append('/home/sr/TRELLIS/llm_prompt_test')

from automated_prompt_generator import PromptGenerator

def test_parsing():
    # Create a test generator with some mock data
    generator = PromptGenerator("/mnt/nas/Benchmark_Datatset/Toys4k/metadata.csv", 3)
    
    # Set up the selected_data to match the original prompts
    generator.selected_data = [
        {'original_caption': 'Pink-brown plane with small wings.', 'sha256': 'test1', 'file_identifier': 'test1.obj'},
        {'original_caption': 'Fan with wooden blades.', 'sha256': 'test2', 'file_identifier': 'test2.obj'},
        {'original_caption': 'Frosted bottle with golden cross.', 'sha256': 'test3', 'file_identifier': 'test3.obj'}
    ]
    
    # Test parsing the qwen output
    qwen_file = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_qwen3_0.6b.txt"
    
    print("Testing qwen3:0.6b parsing...")
    pairs = generator.parse_llm_output(qwen_file)
    
    print(f"\nParsed {len(pairs)} pairs:")
    for i, (original, enhanced) in enumerate(pairs, 1):
        print(f"{i}. Original: '{original}'")
        print(f"   Enhanced: '{enhanced}'")
        print()
    
    # Test parsing the gemma output  
    gemma_file = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    print("Testing gemma3:1b parsing...")
    print("Manually testing pattern 4...")
    
    # Load content and test pattern 4 specifically
    with open(gemma_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    pattern4_pairs = []
    for line in lines:
        line = line.strip()
        # Check if this is a numbered line
        if re.match(r'^(\d+)\.', line):
            print(f"Found numbered line: {repr(line[:50])}")
            
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            if match:
                num_str, rest = match.groups()
                print(f"  num={num_str}, rest start={repr(rest[:10])}, rest end={repr(rest[-10:])}")
                
                # Check quotes
                quote_check = ((rest.startswith('"') and rest.endswith('"')) or
                              (rest.startswith('"') and rest.endswith('"')) or
                              (rest.startswith('"') and rest.endswith('"')))
                print(f"  quote_check={quote_check}")
                quote_chars = ('"', '\u201c', '\u201d')
                print(f"  starts with quote: {rest.startswith(quote_chars)}")
                print(f"  ends with quote: {rest.endswith(quote_chars)}")
                
                if quote_check:
                    enhanced = rest[1:-1]
                    print(f"  enhanced={repr(enhanced[:30])}")
                    num = int(num_str) - 1
                    if num < len(generator.selected_data):
                        original = generator.selected_data[num]['original_caption']
                        pattern4_pairs.append((original.strip(), enhanced.strip()))
                        print(f"  -> Added pair: {original}")
    
    print(f"Pattern 4 found {len(pattern4_pairs)} pairs")
    
    # Now test the actual function
    pairs = generator.parse_llm_output(gemma_file)
    
    print(f"\nActual parsing result: {len(pairs)} pairs:")
    for i, (original, enhanced) in enumerate(pairs, 1):
        print(f"{i}. Original: '{original}'")
        print(f"   Enhanced: '{enhanced}'")
        print()

if __name__ == "__main__":
    test_parsing()