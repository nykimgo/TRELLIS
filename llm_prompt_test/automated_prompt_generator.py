#!/usr/bin/env python3
"""
자동 프롬프트 증강 시스템
Toys4k 데이터셋의 텍스트 프롬프트를 LLM을 사용하여 자동으로 증강하는 스크립트
"""

import csv
import json
import random
import re
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

class PromptGenerator:
    def __init__(self, metadata_path: str, num_samples: int = 100):
        """
        Args:
            metadata_path: metadata.csv 파일 경로
            num_samples: 처리할 랜덤 샘플 수
        """
        self.metadata_path = metadata_path
        self.num_samples = num_samples
        self.selected_data = []
        
        # 출력 디렉토리 생성
        self.output_dir = Path("./prompt_generation_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
    def load_and_sample_data(self) -> List[Dict]:
        """metadata.csv에서 랜덤 샘플링하고 짧은 캡션 선택"""
        print(f"Loading metadata from {self.metadata_path}...")
        
        # CSV 읽기
        df = pd.read_csv(self.metadata_path)
        print(f"Total records: {len(df)}")
        
        # 랜덤 샘플링
        sampled_df = df.sample(n=min(self.num_samples, len(df)), random_state=42)
        print(f"Sampled {len(sampled_df)} records")
        
        selected_data = []
        
        for _, row in sampled_df.iterrows():
            # captions 컬럼에서 짧은 캡션 선택
            captions_str = row['captions']
            
            # NaN/null 값 체크
            if pd.isna(captions_str) or captions_str is None:
                print(f"Skipping row with null captions: {row.get('sha256', 'unknown')}")
                continue
            
            # float 타입 체크 (NaN이 float로 읽힐 수 있음)
            if isinstance(captions_str, float):
                print(f"Skipping row with float captions: {row.get('sha256', 'unknown')}")
                continue
            
            try:
                # JSON 형식의 captions 파싱
                captions = json.loads(captions_str.replace("'", '"'))
                
                # 4-8 단어 길이의 캡션 필터링
                short_captions = [
                    cap for cap in captions 
                    if 4 <= len(cap.split()) <= 8
                ]
                
                if short_captions:
                    # 가장 짧은 캡션 선택
                    selected_caption = min(short_captions, key=lambda x: len(x.split()))
                    
                    selected_data.append({
                        'sha256': row['sha256'],
                        'file_identifier': row['file_identifier'],
                        'original_caption': selected_caption,
                        'all_captions': captions
                    })
                
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                print(f"Error parsing captions for {row.get('sha256', 'unknown')}: {e}")
                continue
        
        self.selected_data = selected_data
        print(f"Successfully selected {len(selected_data)} items with short captions")
        return selected_data
    
    def generate_prompts_file(self, output_file: str = "prompts.txt") -> str:
        """prompts.txt 파일 생성"""
        if not self.selected_data:
            self.load_and_sample_data()
        
        prompt_template = """You are a prompt engineer specialized in converting short user prompts into detailed, descriptive prompts optimized for text-to-3D asset generation models.

Given a short numbered user input list, rewrite each line into a vivid, detailed prompt that:

- Clearly describes the object, scene, or character
- Includes relevant visual details (color, shape, style, posture, materials, environment)
- Avoids ambiguity and is easy for a 3D generation model to interpret
- Uses concise but descriptive language within 40 words or less
- Respond **line by line**, keeping the numbering intact

Please answer in the following format:
"1. <User input>":"<Model response>"

Numbered user input:
"""
        
        # 번호가 매겨진 캡션 리스트 생성
        numbered_captions = []
        for i, data in enumerate(self.selected_data, 1):
            numbered_captions.append(f"{i}. {data['original_caption']}")
        
        # 전체 프롬프트 생성
        full_prompt = prompt_template + "\n".join(numbered_captions)
        
        # 파일 저장
        output_path = self.output_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_prompt)
        
        print(f"Generated prompts file: {output_path}")
        return str(output_path)
    
    def run_ollama_models(self, prompts_file: str, models: List[str]) -> Dict[str, str]:
        """여러 Ollama 모델 실행"""
        results = {}
        
        print(f"Running {len(models)} Ollama models...")
        
        for model in models:
            print(f"Running model: {model}")
            
            # 출력 파일명 생성 (특수문자 제거)
            safe_model_name = re.sub(r'[^\w\-_\.]', '_', model)
            output_file = self.output_dir / f"detailed_prompt_{safe_model_name}.txt"
            
            try:
                # ollama 실행
                cmd = f"ollama run {model} < {prompts_file}"
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=600  # 10분 타임아웃
                )
                
                if result.returncode == 0:
                    # 결과 저장
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.stdout)
                    
                    results[model] = str(output_file)
                    print(f"✓ {model} completed: {output_file}")
                else:
                    print(f"✗ {model} failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print(f"✗ {model} timed out")
            except Exception as e:
                print(f"✗ {model} error: {e}")
        
        return results
    
    def parse_llm_output(self, file_path: str) -> List[Tuple[str, str]]:
        """LLM 출력 파일을 파싱하여 (원본, 증강) 쌍 추출"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 다양한 형식 처리
        pairs = []
        
        # 패턴 1: "번호. 원본":"증강"
        pattern1 = r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"'
        matches1 = re.findall(pattern1, content, re.MULTILINE | re.DOTALL)
        
        for original, enhanced in matches1:
            # 번호 제거
            clean_original = re.sub(r'^\d+\.\s*', '', original).strip()
            enhanced_clean = self._remove_quotes(enhanced)
            pairs.append((clean_original, enhanced_clean))
        
        # 패턴 2: 번호. "원본":"증강" (따옴표 없는 번호)
        if not pairs:
            pattern2 = r'\d+\.\s*"([^"]+)"\s*:\s*"([^"]+)"'
            matches2 = re.findall(pattern2, content, re.MULTILINE | re.DOTALL)
            # 템플릿 텍스트 필터링
            filtered_matches = []
            for orig, enh in matches2:
                if not ('<User input>' in orig or '<Model response>' in enh):
                    filtered_matches.append((orig.strip(), self._remove_quotes(enh)))
            pairs = filtered_matches
        
        # 패턴 3: qwen3 형식 처리 - 번호. "증강된 내용" (ASCII 따옴표)
        if not pairs:
            pattern3 = r'(\d+)\.\s*"([^"]+)"'
            matches3 = re.findall(pattern3, content, re.MULTILINE | re.DOTALL)
            
            if matches3:
                # 원본 데이터와 매칭하기 위해 인덱스 사용
                for i, (num_str, enhanced) in enumerate(matches3):
                    num = int(num_str) - 1  # 0-based index로 변환
                    if num < len(self.selected_data):
                        original = self.selected_data[num]['original_caption']
                        enhanced_clean = self._remove_quotes(enhanced)
                        pairs.append((original.strip(), enhanced_clean))
        
        # 패턴 4: gemma3 형식 처리 - 번호. "증강된 내용" (Unicode 따옴표 지원)
        if not pairs:
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                # 번호로 시작하는 라인 찾기 (중복 번호 처리: "1. 1. content" 형식)
                match = re.match(r'^(\d+)\.\s*(?:\d+\.\s*)?(.+)$', line)
                if match:
                    num_str, rest = match.groups()
                    # 따옴표로 감싸진 내용인지 확인하고 제거 (시작과 끝이 다른 Unicode 따옴표일 수 있음)
                    start_quotes = ('"', '"', '"')  # ASCII, left double, right double
                    end_quotes = ('"', '"', '"')    # ASCII, left double, right double  
                    if (rest.startswith(start_quotes) and rest.endswith(end_quotes)):
                        enhanced = rest[1:-1]  # 첫 번째와 마지막 문자 제거
                        enhanced_clean = self._remove_quotes(enhanced)
                        num = int(num_str) - 1  # 0-based index로 변환
                        if num < len(self.selected_data):
                            original = self.selected_data[num]['original_caption']
                            pairs.append((original.strip(), enhanced_clean))
                    elif not any(char in rest for char in ':'):  # 따옴표 없이 직접 텍스트인 경우 (콜론 없음 확인)
                        enhanced = rest.strip()
                        enhanced_clean = self._remove_quotes(enhanced)
                        num = int(num_str) - 1
                        if num < len(self.selected_data):
                            original = self.selected_data[num]['original_caption']
                            pairs.append((original.strip(), enhanced_clean))
        
        # 패턴 5: 간단한 라인별 처리 (콜론으로 구분)
        if not pairs:
            lines = content.split('\n')
            for line in lines:
                if ':' in line and line.strip():
                    # 번호와 따옴표 제거 후 콜론으로 분할
                    clean_line = re.sub(r'^\d+\.\s*', '', line.strip())
                    clean_line = clean_line.replace('"', '')
                    
                    if ':' in clean_line:
                        parts = clean_line.split(':', 1)
                        if len(parts) == 2:
                            enhanced_clean = self._remove_quotes(parts[1])
                            pairs.append((parts[0].strip(), enhanced_clean))
        
        print(f"Parsed {len(pairs)} prompt pairs from {file_path}")
        return pairs
    
    def create_excel_output_enhanced(self, llm_results: Dict[str, str]) -> str:
        """향상된 통합 Excel 파일 생성"""
        try:
            from enhanced_normalizer import EnhancedNormalizer
            
            # 통합 정규화 실행 (중간 파일들이 저장됨)
            normalizer = EnhancedNormalizer()
            normalized_data = normalizer.normalize_with_best_model(llm_results)
            
            if not normalized_data:
                print("No normalized data available, falling back to legacy method")
                return self.create_excel_output_legacy(llm_results)
            
            # 중복 방지를 위해 각 엔트리에 llm_model이 있는지 확인
            # llm_model이 없거나 잘못된 경우 legacy 방법 사용
            has_valid_source_models = all(
                entry.get('llm_model', 'unknown') != 'unknown' 
                for entry in normalized_data
            )
            
            if not has_valid_source_models:
                print("Source model mapping failed, using legacy method to avoid duplicates...")
                return self.create_excel_output_legacy(llm_results)
            
            # 메타데이터 딕셔너리 생성
            metadata_dict = {data['original_caption']: data for data in self.selected_data}
            
            # 통합 Excel 파일 생성
            excel_path = self.output_dir / "automated_prompt_results.xlsx"
            result_path = normalizer.create_unified_excel(normalized_data, metadata_dict, str(excel_path), llm_results)
            
            return result_path
            
        except Exception as e:
            print(f"Enhanced Excel creation failed: {e}")
            print("Falling back to legacy method...")
            return self.create_excel_output_legacy(llm_results)

    def create_excel_output_legacy(self, llm_results: Dict[str, str]) -> str:
        """기존 Excel 파일 생성 방식 (fallback)"""
        all_rows = []
        object_name_cache = {}  # 원본 프롬프트별로 object_name 캐시
        metadata_cache = {}  # 원본 프롬프트별로 metadata 캐시
        
        # 첫 번째 패스: 모든 모델의 결과를 파싱하고 캐시 구축
        for model, file_path in llm_results.items():
            try:
                pairs = self.parse_llm_output(file_path)
                
                for i, (original, enhanced) in enumerate(pairs):
                    original_lower = original.strip().lower()
                    
                    # 원본 데이터에서 매칭되는 항목 찾기
                    matched_data = None
                    for data in self.selected_data:
                        if data['original_caption'].strip().lower() == original_lower:
                            matched_data = data
                            break
                    
                    # metadata 캐시에 저장 (매칭된 데이터가 있으면)
                    if matched_data and original_lower not in metadata_cache:
                        metadata_cache[original_lower] = matched_data
                    
                    # object_name 추출 및 캐시 (항상 시도, Unknown이 아닌 경우만 캐시)
                    if original_lower not in object_name_cache:
                        object_name = self._extract_object_name(original)
                        if object_name != "Unknown":
                            object_name_cache[original_lower] = object_name
                            
            except Exception as e:
                print(f"Error in first pass processing {model}: {e}")
                continue
        
        # 두 번째 패스: 실제 행 생성 (캐시 활용)
        for model, file_path in llm_results.items():
            try:
                pairs = self.parse_llm_output(file_path)
                
                # 모델 정보 추출
                model_parts = model.split(':')
                model_name = model_parts[0]
                model_size = model_parts[1] if len(model_parts) > 1 else "unknown"
                
                for i, (original, enhanced) in enumerate(pairs):
                    original_lower = original.strip().lower()
                    
                    # 캐시에서 object_name 가져오기
                    object_name = object_name_cache.get(original_lower, "Unknown")
                    
                    # Unknown인 경우 캐시에서 유사한 프롬프트의 좋은 object_name 찾기
                    if object_name == "Unknown":
                        for cached_prompt, cached_name in object_name_cache.items():
                            if self._prompts_similar(original_lower, cached_prompt) and cached_name != "Unknown":
                                object_name = cached_name
                                print(f"Shared object_name '{cached_name}' from similar prompt for '{original[:30]}...'")
                                break
                    
                    # 캐시에서 metadata 가져오기
                    matched_data = metadata_cache.get(original_lower, None)
                    
                    # enhanced text 처리 - 숫자 제거 및 따옴표 추가
                    enhanced_clean = self._clean_enhanced_text(enhanced, model_name)
                    
                    row = {
                        'category': model_name,
                        'llm_model': model,
                        'parameters': model_size,
                        'size': 'unknown',
                        'GPU_usage': 'unknown',
                        'object_name': object_name,
                        'seed': '',
                        'params': '',
                        'matched_image': '',
                        'object_name_clean': object_name,
                        'user_prompt': original,
                        'text_prompt': enhanced_clean,
                        'sha256': matched_data['sha256'] if matched_data else '',
                        'file_identifier': matched_data['file_identifier'] if matched_data else ''
                    }
                    all_rows.append(row)
                    
            except Exception as e:
                print(f"Error processing {model}: {e}")
                continue
        
        # DataFrame 생성 및 Excel 저장
        df = pd.DataFrame(all_rows)
        excel_path = self.output_dir / "automated_prompt_results.xlsx"
        df.to_excel(excel_path, index=False)
        
        print(f"Legacy Excel file created: {excel_path}")
        print(f"Total rows: {len(all_rows)}")
        
        return str(excel_path)
    
    def _remove_quotes(self, text: str) -> str:
        """텍스트에서 시작과 끝의 쌍따옴표 제거 (일관성을 위해)"""
        text = text.strip()
        quote_chars = ('"', '"', '"')  # ASCII 및 Unicode 쌍따옴표
        
        # 시작과 끝이 모두 쌍따옴표인 경우 제거
        if text.startswith(quote_chars) and text.endswith(quote_chars):
            return text[1:-1].strip()
        
        return text
    
    def _extract_object_name(self, prompt: str) -> str:
        """간단한 객체명 추출 (나중에 더 정교한 LLM으로 대체 가능)"""
        # 간단한 키워드 추출
        words = prompt.lower().split()
        
        # 일반적인 객체 키워드들
        object_keywords = [
            'table', 'chair', 'bed', 'lamp', 'book', 'cup', 'bowl', 'plate', 
            'phone', 'television', 'computer', 'car', 'house', 'tree', 'flower',
            'cabinet', 'drawer', 'shelf', 'mirror', 'clock', 'bottle', 'box',
            'plane', 'airplane', 'fan', 'frosted'
        ]
        
        for word in words:
            if word in object_keywords:
                return word.capitalize()
        
        # 특별한 경우 처리
        if 'plane' in prompt.lower() or 'airplane' in prompt.lower():
            return 'Pink-brown'
        if 'fan' in prompt.lower() and 'wooden' in prompt.lower():
            return 'Fan'
        if 'bottle' in prompt.lower() and ('frosted' in prompt.lower() or 'cross' in prompt.lower()):
            return 'Bottle'
        
        # 첫 번째 명사로 추정되는 단어 반환
        return words[0].capitalize() if words else "Unknown"
    
    def _clean_enhanced_text(self, text: str, model_name: str) -> str:
        """enhanced text 정리 - 숫자 제거 및 따옴표 추가"""
        # 기본 정리
        cleaned = text.strip()
        
        # gemma3의 경우 앞의 숫자 제거 (예: "1. A petite..." -> "A petite...")
        if model_name == 'gemma3':
            cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
        
        # qwen3와 deepseek의 경우 따옴표 추가
        if model_name in ['qwen3', 'deepseek-r1']:
            # 이미 따옴표가 있으면 제거 후 다시 추가
            cleaned = cleaned.strip('"\'')
            cleaned = f'"{cleaned}"'
        
        return cleaned
    
    def _prompts_similar(self, prompt1: str, prompt2: str) -> bool:
        """두 프롬프트의 유사도 확인"""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union
        
        return similarity > 0.7  # 70% 이상 유사
    
    def run_full_pipeline(self, models: List[str]) -> str:
        """전체 파이프라인 실행"""
        print("=== Starting Automated Prompt Generation Pipeline ===")
        
        # 1. 데이터 로드 및 샘플링
        self.load_and_sample_data()
        
        # 2. prompts.txt 생성
        prompts_file = self.generate_prompts_file()
        
        # 3. Ollama 모델들 실행
        llm_results = self.run_ollama_models(prompts_file, models)
        
        if not llm_results:
            print("No successful LLM results. Aborting.")
            return ""
        
        # 4. 향상된 Excel 파일 생성 (통합 정규화 + 객체명 생성)
        excel_path = self.create_excel_output_enhanced(llm_results)
        
        print("=== Pipeline Completed ===")
        print(f"Results saved to: {excel_path}")
        print(f"Output directory: {self.output_dir}")
        
        return excel_path

def main():
    """메인 실행 함수"""
    # 기본 설정
    metadata_path = "datasets/HSSD/metadata.csv"
    num_samples = 100
    
    # 사용할 Ollama 모델들 (사용자가 설치된 모델로 수정 필요)
    models = [
        "gemma3:1b",
        "gemma3:12b", 
        "qwen3:0.6b",
        "qwen3:1.7b",
        "qwen3:14b",
        "deepseek-r1:1.5b"
    ]
    
    # 커맨드라인 인자 처리
    if len(sys.argv) > 1:
        metadata_path = sys.argv[1]
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])
    
    print(f"Metadata path: {metadata_path}")
    print(f"Number of samples: {num_samples}")
    print(f"Models to use: {models}")
    
    # 파이프라인 실행
    generator = PromptGenerator(metadata_path, num_samples)
    result_path = generator.run_full_pipeline(models)
    
    if result_path:
        print(f"\n✓ Success! Results saved to: {result_path}")
    else:
        print("\n✗ Pipeline failed!")

if __name__ == "__main__":
    main()