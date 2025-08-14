#!/usr/bin/env python3
"""
Debug multiline content matching
"""

import re

def debug_multiline():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Try the pattern on the full content
    pattern = r'(\d+)\.\s*"([^"]+)"'
    
    print("=== Testing on full content ===")
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    print(f"Findall matches: {matches}")
    
    # Try re.search to see if any match exists
    search_result = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    print(f"Search result: {search_result}")
    if search_result:
        print(f"Search groups: {search_result.groups()}")
    
    # Try finditer to see all matches
    print("\n=== Using finditer ===")
    for i, match in enumerate(re.finditer(pattern, content, re.MULTILINE | re.DOTALL)):
        print(f"Match {i+1}: {match.groups()}")
        print(f"  Span: {match.span()}")
        print(f"  Text: {repr(content[match.start():match.end()])}")
    
    # Let's also test on a reconstructed version
    print("\n=== Testing on reconstructed content ===")
    lines = content.split('\n')
    target_lines = []
    for line in lines:
        if re.match(r'^\d+\.', line.strip()):
            target_lines.append(line.strip())
    
    reconstructed = '\n'.join(target_lines)
    print(f"Reconstructed: {repr(reconstructed)}")
    
    matches_reconstructed = re.findall(pattern, reconstructed, re.MULTILINE | re.DOTALL)
    print(f"Matches on reconstructed: {matches_reconstructed}")

if __name__ == "__main__":
    debug_multiline()