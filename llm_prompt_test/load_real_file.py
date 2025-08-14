#!/usr/bin/env python3
"""
Load the real file and check characters
"""

import re

def load_real_file():
    file_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_gemma3_1b.txt"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the first numbered line
    for line in lines:
        if '1.' in line:  # Just look for numbered lines
            line = line.strip()
            print(f"Found line: {repr(line)}")
            print(f"Length: {len(line)}")
            
            # Find all quote-like characters
            quotes_found = []
            for i, char in enumerate(line):
                if ord(char) in [34, 8220, 8221, 8216, 8217]:  # Various quote types
                    quotes_found.append((i, char, ord(char)))
            
            print(f"Quote characters found: {quotes_found}")
            
            if quotes_found:
                # Extract content between first and last quote
                first_quote_pos = quotes_found[0][0]
                last_quote_pos = quotes_found[-1][0]
                content = line[first_quote_pos+1:last_quote_pos]
                print(f"Content between quotes: {repr(content)}")
                
                # Try simple extraction with any quote
                match = re.match(r'^(\d+)\.\s*(.+)$', line)
                if match:
                    num, rest = match.groups()
                    print(f"Number: {num}")
                    print(f"Rest: {repr(rest)}")
                    
                    # Remove quotes from the beginning and end
                    if rest.startswith(('"', '"', '"')) and rest.endswith(('"', '"', '"')):
                        clean_content = rest[1:-1]
                        print(f"Clean content: {repr(clean_content)}")
            break

if __name__ == "__main__":
    load_real_file()