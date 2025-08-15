#\!/usr/bin/env python3
import re
from pathlib import Path

def debug_parse_llm_output(file_path: str):
    """디버깅용 LLM 출력 파싱"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"=== 파일 내용 ({file_path}) ===")
    print(content[:500] + "..." if len(content) > 500 else content)
    print()
    
    pairs = []
    
    # 패턴 1: "번호. 원본":"증강"
    print("=== 패턴 1 테스트: \"번호. 원본\":\"증강\" ===")
    pattern1 = r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"'
    matches1 = re.findall(pattern1, content, re.MULTILINE | re.DOTALL)
    print(f"매치 수: {len(matches1)}")
    
    if matches1:
        for original, enhanced in matches1:
            clean_original = re.sub(r'^\d+\.\s*', '', original).strip()
            pairs.append((clean_original, enhanced))
            print(f"  원본: '{original}' -> 정리: '{clean_original}'")
            print(f"  증강: '{enhanced}'")
    
    # 패턴 2: 번호. "원본":"증강"
    if not pairs:
        print("\n=== 패턴 2 테스트: 번호. \"원본\":\"증강\" ===")
        pattern2 = r'\d+\.\s*"([^"]+)"\s*:\s*"([^"]+)"'
        matches2 = re.findall(pattern2, content, re.MULTILINE | re.DOTALL)
        print(f"매치 수: {len(matches2)}")
        
        if matches2:
            for orig, enh in matches2:
                pairs.append((orig.strip(), enh))
                print(f"  원본: '{orig.strip()}'")
                print(f"  증강: '{enh}'")
    
    print(f"\n=== 최종 파싱 결과: {len(pairs)} 쌍 ===")
    for i, (orig, enh) in enumerate(pairs):
        print(f"{i+1}. '{orig}' -> '{enh[:50]}...'")
    
    return pairs

# qwen3 파일 파싱 테스트
print("qwen3 파일 파싱 테스트:")
qwen3_pairs = debug_parse_llm_output("prompt_generation_outputs/detailed_prompt_qwen3_0.6b.txt")

print("\n" + "="*80)
print("gemma3 파일 파싱 테스트:")
gemma3_pairs = debug_parse_llm_output("prompt_generation_outputs/detailed_prompt_gemma3_1b.txt")
EOF < /dev/null
