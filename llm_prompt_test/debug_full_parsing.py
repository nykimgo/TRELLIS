#!/usr/bin/env python3
"""
Debug the full parsing process step by step
"""

import re

def debug_full_parsing():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pairs = []
    
    print("=== Testing Pattern 1 ===")
    pattern1 = r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"'
    matches1 = re.findall(pattern1, content, re.MULTILINE | re.DOTALL)
    print(f"Pattern 1 matches: {matches1}")
    
    for original, enhanced in matches1:
        clean_original = re.sub(r'^\d+\.\s*', '', original).strip()
        pairs.append((clean_original, enhanced.strip()))
    
    print(f"After pattern 1, pairs count: {len(pairs)}")
    
    if not pairs:
        print("=== Testing Pattern 2 ===")
        pattern2 = r'\d+\.\s*"([^"]+)"\s*:\s*"([^"]+)"'
        matches2 = re.findall(pattern2, content, re.MULTILINE | re.DOTALL)
        print(f"Pattern 2 matches: {matches2}")
        pairs = [(orig.strip(), enh.strip()) for orig, enh in matches2]
        print(f"After pattern 2, pairs count: {len(pairs)}")
    
    if not pairs:
        print("=== Testing Pattern 3 (qwen format) ===")
        pattern3 = r'(\d+)\.\s*"([^"]+)"'
        matches3 = re.findall(pattern3, content, re.MULTILINE | re.DOTALL)
        print(f"Pattern 3 matches: {matches3}")
        # This would need selected_data, but let's see if matches are found
        print(f"Pattern 3 would create {len(matches3)} pairs")
    
    if not pairs:
        print("=== Testing Pattern 4 (gemma line by line) ===")
        lines = content.split('\n')
        line_pairs = []
        for line in lines:
            line = line.strip()
            match = re.match(r'^(\d+)\.\s*"(.+)"', line)
            if match:
                num_str, enhanced = match.groups()
                print(f"Line matched: {num_str} -> {enhanced}")
                # Would need selected_data here too
                line_pairs.append((f"original_{num_str}", enhanced.strip()))
        print(f"Pattern 4 found {len(line_pairs)} line matches")
    
    if not pairs:
        print("=== Testing Pattern 5 (colon separated) ===")
        lines = content.split('\n')
        for line in lines:
            if ':' in line and line.strip():
                print(f"Line with colon: {repr(line)}")
                clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
                clean_line = clean_line.replace('"', '')
                print(f"  Cleaned: {repr(clean_line)}")
                
                if ':' in clean_line:
                    parts = clean_line.split(':', 1)
                    if len(parts) == 2:
                        print(f"  Would add pair: ({repr(parts[0].strip())}, {repr(parts[1].strip())})")
                        pairs.append((parts[0].strip(), parts[1].strip()))
    
    print(f"Final pairs count: {len(pairs)}")
    for i, (orig, enh) in enumerate(pairs):
        print(f"  {i+1}. '{orig}' -> '{enh}'")

if __name__ == "__main__":
    debug_full_parsing()