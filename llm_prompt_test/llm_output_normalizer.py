#!/usr/bin/env python3
"""
LLM 출력 정규화 도구
다양한 LLM 모델의 출력 형식을 통일된 형태로 변환
"""

import re
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
import subprocess

class LLMOutputNormalizer:
    def __init__(self):
        self.normalization_prompt = """
You are a text processing assistant. Your task is to convert various LLM output formats into a standardized format.

Given an input text that contains original prompts and their enhanced versions in various formats, convert it to the following exact format:

"<original_prompt>":"<enhanced_prompt>"

Rules:
1. Extract each original-enhanced prompt pair
2. Remove any numbering (1., 2., etc.)
3. Clean up extra whitespace and formatting
4. Output ONLY the pairs in the specified format, one per line
5. Do not add any explanations or headers

Input text:
"""

    def normalize_with_ollama(self, input_file: str, model: str = "qwen3:1.7b") -> str:
        """Ollama를 사용하여 출력 정규화"""
        print(f"Normalizing {input_file} with {model}")
        
        # 입력 파일 읽기
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 정규화 프롬프트 생성
        full_prompt = self.normalization_prompt + content
        
        # 임시 프롬프트 파일 생성
        temp_prompt_file = Path(input_file).parent / "temp_normalize_prompt.txt"
        with open(temp_prompt_file, 'w', encoding='utf-8') as f:
            f.write(full_prompt)
        
        try:
            # Ollama 실행
            cmd = f"ollama run {model} < {temp_prompt_file}"
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                # 정규화된 출력 저장
                normalized_file = input_file.replace('.txt', '_normalized.txt')
                with open(normalized_file, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                
                print(f"✓ Normalized output saved to: {normalized_file}")
                return normalized_file
            else:
                print(f"✗ Normalization failed: {result.stderr}")
                return ""
                
        except Exception as e:
            print(f"✗ Error during normalization: {e}")
            return ""
        finally:
            # 임시 파일 삭제
            if temp_prompt_file.exists():
                temp_prompt_file.unlink()
    
    def normalize_manually(self, input_file: str) -> str:
        """수동 정규화 (rule-based)"""
        print(f"Manually normalizing {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        normalized_pairs = []
        
        # 여러 패턴 시도
        patterns = [
            # 패턴 1: "번호. 원본":"증강"
            r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"',
            # 패턴 2: 번호. "원본":"증강"
            r'\d+\.\s*"([^"]+)"\s*:\s*"([^"]+)"',
            # 패턴 3: 번호. 원본: 증강
            r'\d+\.\s*([^:]+):\s*([^\n]+)',
            # 패턴 4: "원본":"증강" (번호 없음)
            r'"([^"]+)"\s*:\s*"([^"]+)"'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches:
                print(f"Found {len(matches)} matches with pattern: {pattern}")
                
                for match in matches:
                    original, enhanced = match
                    
                    # 번호 제거
                    clean_original = re.sub(r'^\d+\.\s*', '', original).strip()
                    clean_original = clean_original.strip('"\'')
                    
                    # 불필요한 문자 제거
                    clean_enhanced = enhanced.strip().strip('"\'')
                    
                    if clean_original and clean_enhanced:
                        normalized_pairs.append(f'"{clean_original}":"{clean_enhanced}"')
                
                break  # 첫 번째 성공한 패턴 사용
        
        if not normalized_pairs:
            print("No patterns matched, trying line-by-line processing")
            # 라인별 처리
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line and len(line) > 10:  # 최소 길이 체크
                    # 번호 제거
                    clean_line = re.sub(r'^\d+\.\s*', '', line)
                    
                    if ':' in clean_line:
                        parts = clean_line.split(':', 1)
                        if len(parts) == 2:
                            original = parts[0].strip().strip('"\'')
                            enhanced = parts[1].strip().strip('"\'')
                            
                            if original and enhanced:
                                normalized_pairs.append(f'"{original}":"{enhanced}"')
        
        # 정규화된 출력 저장
        if normalized_pairs:
            normalized_file = input_file.replace('.txt', '_normalized.txt')
            with open(normalized_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(normalized_pairs))
            
            print(f"✓ Manually normalized {len(normalized_pairs)} pairs to: {normalized_file}")
            return normalized_file
        else:
            print("✗ Manual normalization failed - no pairs found")
            return ""

def normalize_all_outputs(output_dir: str, use_ollama: bool = True):
    """출력 디렉토리의 모든 LLM 결과 파일을 정규화"""
    normalizer = LLMOutputNormalizer()
    output_path = Path(output_dir)
    
    # detailed_prompt_로 시작하는 모든 파일 찾기
    llm_files = list(output_path.glob("detailed_prompt_*.txt"))
    
    if not llm_files:
        print("No LLM output files found to normalize")
        return
    
    print(f"Found {len(llm_files)} LLM output files to normalize")
    
    for file_path in llm_files:
        if "_normalized" in str(file_path):
            continue  # 이미 정규화된 파일은 스킵
        
        try:
            if use_ollama:
                # Ollama를 사용한 정규화
                result = normalizer.normalize_with_ollama(str(file_path))
                if not result:
                    # Ollama 실패시 수동 정규화로 fallback
                    print("Ollama normalization failed, trying manual normalization...")
                    normalizer.normalize_manually(str(file_path))
            else:
                # 수동 정규화만 사용
                normalizer.normalize_manually(str(file_path))
                
        except Exception as e:
            print(f"Error normalizing {file_path}: {e}")

if __name__ == "__main__":
    import sys
    
    output_dir = "prompt_generation_outputs"
    use_ollama = True
    
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    if len(sys.argv) > 2:
        use_ollama = sys.argv[2].lower() in ['true', '1', 'yes']
    
    print(f"Normalizing outputs in: {output_dir}")
    print(f"Use Ollama for normalization: {use_ollama}")
    
    normalize_all_outputs(output_dir, use_ollama)