#!/usr/bin/env python3
import re

# qwen3 실제 출력 내용
qwen3_content = '''1. "Pink-brown plane with small wings":"A pink-brown wooden plane with delicate, butterfly-like wings, casting a soft glow over a forest canopy."  
2. "Fan with wooden blades":"A wooden fan with ornate blades, casting a golden light over a sunny garden."  
3. "Frosted bottle with golden cross":"A frosted bottle with a radiant golden cross, holding a warm, nostalgic glow inside."'''

print("=== qwen3 내용 ===")
print(qwen3_content)
print()

print("=== 패턴 테스트 ===")

# 패턴 1: "번호. 원본":"증강"
pattern1 = r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"'
matches1 = re.findall(pattern1, qwen3_content, re.MULTILINE | re.DOTALL)
print(f"패턴 1 결과: {len(matches1)} matches")
for i, (orig, enh) in enumerate(matches1):
    print(f"  {i+1}: '{orig}' -> '{enh}'")

print()

# 패턴 2: 번호. "원본":"증강"
pattern2 = r'\d+\.\s*"([^"]+)"\s*:\s*"([^"]+)"'
matches2 = re.findall(pattern2, qwen3_content, re.MULTILINE | re.DOTALL)
print(f"패턴 2 결과: {len(matches2)} matches")
for i, (orig, enh) in enumerate(matches2):
    print(f"  {i+1}: '{orig}' -> '{enh}'")

print()

# 수정된 패턴 테스트
# 번호가 따옴표 밖에 있는 경우
pattern_fixed = r'(\d+)\.\s*"([^"]+)"\s*:\s*"([^"]+)"'
matches_fixed = re.findall(pattern_fixed, qwen3_content, re.MULTILINE | re.DOTALL)
print(f"수정된 패턴 결과: {len(matches_fixed)} matches")
for i, (num, orig, enh) in enumerate(matches_fixed):
    print(f"  {i+1}: {num}. '{orig}' -> '{enh}'")