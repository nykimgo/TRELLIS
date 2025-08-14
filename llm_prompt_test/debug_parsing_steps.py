#!/usr/bin/env python3
"""
Debug the parsing steps in detail
"""

import sys
import os
sys.path.append('/home/sr/TRELLIS/llm_prompt_test')

from automated_prompt_generator import PromptGenerator
import re

def debug_parsing_steps():
    # Create a test generator with some mock data
    generator = PromptGenerator("/mnt/nas/Benchmark_Datatset/Toys4k/metadata.csv", 3)
    
    # Set up the selected_data to match the original prompts
    generator.selected_data = [
        {'original_caption': 'Pink-brown plane with small wings.', 'sha256': 'test1', 'file_identifier': 'test1.obj'},
        {'original_caption': 'Fan with wooden blades.', 'sha256': 'test2', 'file_identifier': 'test2.obj'},
        {'original_caption': 'Frosted bottle with golden cross.', 'sha256': 'test3', 'file_identifier': 'test3.obj'}
    ]
    
    # Load the gemma file content
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Content length: {len(content)}")
    print(f"Content preview: {repr(content[:200])}")
    print()
    
    # Manually reproduce the parsing logic with debug output
    pairs = []
    
    print("=== Pattern 1 Test ===")
    pattern1 = r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"'
    matches1 = re.findall(pattern1, content, re.MULTILINE | re.DOTALL)
    print(f"Pattern 1 matches: {matches1}")
    
    for original, enhanced in matches1:
        clean_original = re.sub(r'^\d+\.\s*', '', original).strip()
        pairs.append((clean_original, enhanced.strip()))
    print(f"Pairs after pattern 1: {len(pairs)}")
    
    if not pairs:
        print("\n=== Pattern 2 Test ===")
        pattern2 = r'\d+\.\s*"([^"]+)"\s*:\s*"([^"]+)"'
        matches2 = re.findall(pattern2, content, re.MULTILINE | re.DOTALL)
        print(f"Pattern 2 matches: {matches2}")
        pairs = [(orig.strip(), enh.strip()) for orig, enh in matches2]
        print(f"Pairs after pattern 2: {len(pairs)}")
    
    if not pairs:
        print("\n=== Pattern 3 Test ===")
        pattern3 = r'(\d+)\.\s*"([^"]+)"'
        matches3 = re.findall(pattern3, content, re.MULTILINE | re.DOTALL)
        print(f"Pattern 3 matches: {matches3}")
        print(f"Selected data count: {len(generator.selected_data)}")
        
        if matches3:
            for i, (num_str, enhanced) in enumerate(matches3):
                num = int(num_str) - 1
                print(f"  Match {i}: num_str={num_str}, num={num}, enhanced={enhanced[:50]}...")
                if num < len(generator.selected_data):
                    original = generator.selected_data[num]['original_caption']
                    pairs.append((original.strip(), enhanced.strip()))
                    print(f"    -> Added pair: {original}")
        print(f"Pairs after pattern 3: {len(pairs)}")
    
    if not pairs:
        print("\n=== Pattern 4 Test (Unicode quotes) ===")
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.', line):
                print(f"Testing line: {repr(line[:50])}")
                match = re.match(r'^(\d+)\.\s*[""](.+?)[""]', line)
                if match:
                    num_str, enhanced = match.groups()
                    print(f"  ✓ Unicode match: num={num_str}, enhanced={enhanced[:50]}...")
                    num = int(num_str) - 1
                    if num < len(generator.selected_data):
                        original = generator.selected_data[num]['original_caption']
                        pairs.append((original.strip(), enhanced.strip()))
                        print(f"    -> Added pair: {original}")
                else:
                    print(f"  ✗ No Unicode match")
        print(f"Pairs after pattern 4: {len(pairs)}")
    
    print(f"\nFinal result: {len(pairs)} pairs")
    for i, (original, enhanced) in enumerate(pairs):
        print(f"{i+1}. '{original}' -> '{enhanced[:50]}...'")

if __name__ == "__main__":
    debug_parsing_steps()