#!/usr/bin/env python3
"""
향상된 LLM 출력 정규화 도구
JSON 형식으로 통합된 정규화 및 객체명 생성
"""

import re
import json
import os
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

class EnhancedNormalizer:
    def __init__(self):
        self.unified_prompt = """
You are an expert text processing assistant. Your task is to process LLM-generated prompt enhancement outputs and extract structured information.

Given raw LLM output containing original prompts and their enhanced versions, extract and return the data in this exact JSON format for each pair:

{"object_name": "<main_object>", "user_prompt": "<original_short_prompt>", "text_prompt": "<enhanced_detailed_prompt>"}

Rules:
1. object_name: Extract the main object (1-3 words, singular form, no articles)
2. user_prompt: Original short prompt (clean, no numbering)  
3. text_prompt: Enhanced detailed prompt (clean, descriptive)
4. Return ONLY valid JSON objects, one per line
5. No explanations, headers, or additional text

Examples:
Input: 1. "Dark wooden table":"A dark wooden side table with carved legs..."
Output: {"object_name": "Table", "user_prompt": "Dark wooden table", "text_prompt": "A dark wooden side table with carved legs and a smooth surface finish"}

Input: 2. "Red sports car":"A sleek red sports car with aerodynamic design..."  
Output: {"object_name": "Car", "user_prompt": "Red sports car", "text_prompt": "A sleek red sports car with aerodynamic design and chrome details"}

Now process this input:

"""

    def get_best_model(self, available_models: List[str]) -> str:
        """사용 가능한 모델 중에서 파라미터 크기가 가장 큰 모델 선택"""
        def extract_params(model_name):
            try:
                if ':' in model_name:
                    param_part = model_name.split(':', 1)[1]
                    match = re.search(r'(\d+(?:\.\d+)?)b', param_part, re.IGNORECASE)
                    if match:
                        return float(match.group(1))
            except:
                pass
            return 0
        
        if not available_models:
            return None
            
        best_model = max(available_models, key=extract_params)
        best_params = extract_params(best_model)
        
        print(f"Selected best normalization model: {best_model} ({best_params}B params)")
        return best_model

    def normalize_with_best_model(self, llm_results: Dict[str, str]) -> List[Dict]:
        """최고 성능 모델로 모든 LLM 출력을 정규화 (외부 API 우선)"""
        
        # 1. 외부 API 우선 시도
        try:
            from external_api_normalizer import ExternalAPINormalizer
            
            api_normalizer = ExternalAPINormalizer()
            
            # 활성화된 API가 있는지 확인
            has_enabled_api = any(api["enabled"] for api in [
                api_normalizer.config["groq"], 
                api_normalizer.config["openai"], 
                api_normalizer.config["claude"], 
                api_normalizer.config["gemini"]
            ])
            
            if has_enabled_api:
                print("🚀 Trying external API for fast normalization...")
                result = api_normalizer.normalize_with_external_api(llm_results)
                if result:
                    return result
            else:
                print("No external APIs configured, falling back to local Ollama")
                
        except Exception as e:
            print(f"External API failed: {e}")
            print("Falling back to local Ollama...")
        
        # 2. 로컬 Ollama 모델로 fallback
        return self.normalize_with_ollama(llm_results)

    def normalize_with_ollama(self, llm_results: Dict[str, str]) -> List[Dict]:
        """로컬 Ollama 모델로 정규화"""
        
        # 사용 가능한 모델 확인
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode != 0:
                print("Error: Cannot access Ollama")
                return self.fallback_normalize(llm_results)
            
            available_models = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    available_models.append(model_name)
                    
        except Exception as e:
            print(f"Error checking models: {e}")
            return self.fallback_normalize(llm_results)
        
        # 최고 모델 선택
        best_model = self.get_best_model(available_models)
        if not best_model:
            print("No suitable model found, using fallback normalization")
            return self.fallback_normalize(llm_results)
        
        print(f"Using local Ollama model: {best_model}")
        
        # 모든 LLM 출력을 하나로 합치기
        combined_content = ""
        for model, file_path in llm_results.items():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                combined_content += f"\n=== Output from {model} ===\n{content}\n"
        
        # 통합 프롬프트 생성
        full_prompt = self.unified_prompt + combined_content
        
        # 임시 파일 생성
        temp_prompt_file = Path(list(llm_results.values())[0]).parent / "temp_unified_prompt.txt"
        with open(temp_prompt_file, 'w', encoding='utf-8') as f:
            f.write(full_prompt)
        
        try:
            # 최고 모델로 정규화 실행
            cmd = f"ollama run {best_model} < {temp_prompt_file}"
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10분 타임아웃
            )
            
            if result.returncode == 0:
                # JSON 출력 파싱
                normalized_data = self.parse_json_output(result.stdout)
                
                # 결과 저장
                output_file = temp_prompt_file.parent / f"unified_normalized_{best_model.replace(':', '_')}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.stdout)
                
                print(f"✓ Unified normalization completed: {len(normalized_data)} entries")
                print(f"✓ Raw output saved to: {output_file}")
                
                return normalized_data
            else:
                print(f"✗ Ollama normalization failed: {result.stderr}")
                return self.fallback_normalize(llm_results)
                
        except subprocess.TimeoutExpired:
            print("✗ Ollama normalization timed out")
            return self.fallback_normalize(llm_results)
        except Exception as e:
            print(f"✗ Error during Ollama normalization: {e}")
            return self.fallback_normalize(llm_results)
        finally:
            # 임시 파일 정리
            if temp_prompt_file.exists():
                temp_prompt_file.unlink()

    def parse_json_output(self, output: str) -> List[Dict]:
        """JSON 출력 파싱"""
        normalized_data = []
        
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and line.startswith('{') and line.endswith('}'):
                try:
                    data = json.loads(line)
                    
                    # 필수 필드 확인
                    if all(key in data for key in ['object_name', 'user_prompt', 'text_prompt']):
                        normalized_data.append({
                            'object_name': str(data['object_name']).strip(),
                            'user_prompt': str(data['user_prompt']).strip(),
                            'text_prompt': str(data['text_prompt']).strip()
                        })
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"Successfully parsed {len(normalized_data)} JSON entries")
        return normalized_data

    def fallback_normalize(self, llm_results: Dict[str, str]) -> List[Dict]:
        """LLM 실패시 rule-based 정규화"""
        print("Using fallback rule-based normalization")
        
        from object_name_generator import ObjectNameGenerator
        obj_generator = ObjectNameGenerator()
        
        normalized_data = []
        
        for model, file_path in llm_results.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 기존 정규화 패턴들
                patterns = [
                    r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"',
                    r'\d+\.\s*"([^"]+)"\s*:\s*"([^"]+)"',
                    r'\d+\.\s*([^:]+):\s*([^\n]+)',
                    r'"([^"]+)"\s*:\s*"([^"]+)"'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                    if matches:
                        for original, enhanced in matches:
                            clean_original = re.sub(r'^\d+\.\s*', '', original).strip().strip('"\'')
                            clean_enhanced = enhanced.strip().strip('"\'')
                            
                            if clean_original and clean_enhanced:
                                object_name = obj_generator._fallback_object_extraction(clean_original)
                                
                                normalized_data.append({
                                    'object_name': object_name,
                                    'user_prompt': clean_original,
                                    'text_prompt': clean_enhanced
                                })
                        break
                        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Fallback normalization completed: {len(normalized_data)} entries")
        return normalized_data

    def create_unified_excel(self, normalized_data: List[Dict], metadata_dict: Dict, output_path: str, llm_results: Dict[str, str] = None) -> str:
        """통합된 Excel 파일 생성 - 모델별로 중복 생성"""
        
        # 원본 LLM 결과에서 모델별 프롬프트 매핑 생성
        model_prompt_map = {}
        if llm_results:
            print(f"Creating model mapping from {len(llm_results)} LLM result files...")
            
            for model, file_path in llm_results.items():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    print(f"Parsing {model} results...")
                    
                    # 기본 패턴으로 파싱하여 해당 모델의 프롬프트들 추출
                    patterns = [
                        r'"(\d+\.\s*[^"]+)"\s*:\s*"([^"]+)"',
                        r'\d+\.\s*"([^"]+)"\s*:\s*"([^"]+)"',
                        r'\d+\.\s*([^:]+):\s*([^\n]+)',
                        r'"([^"]+)"\s*:\s*"([^"]+)"'
                    ]
                    
                    found_matches = 0
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                        if matches:
                            for original, enhanced in matches:
                                clean_original = re.sub(r'^\d+\.\s*', '', original).strip().strip('"\'').lower()
                                model_prompt_map[clean_original] = model
                                found_matches += 1
                            break
                    
                    print(f"  Found {found_matches} prompts from {model}")
                            
                except Exception as e:
                    print(f"Warning: Could not parse {model} for model mapping: {e}")
            
            print(f"Total mapped prompts: {len(model_prompt_map)}")
            if model_prompt_map:
                print("Sample mappings:")
                for i, (prompt, model) in enumerate(list(model_prompt_map.items())[:3]):
                    print(f"  '{prompt[:30]}...' → {model}")
        
        # 메타데이터와 매칭
        for entry in normalized_data:
            user_prompt = entry['user_prompt'].lower().strip()
            
            # 원본 데이터에서 매칭 시도
            for original_data in metadata_dict.values():
                if original_data['original_caption'].lower().strip() == user_prompt:
                    entry.update({
                        'sha256': original_data['sha256'],
                        'file_identifier': original_data['file_identifier']
                    })
                    break
            else:
                # 매칭 실패시 빈 값
                entry.update({
                    'sha256': '',
                    'file_identifier': ''
                })
            
            # LLM 모델 정보 매칭 (정확 매칭 우선, 그 다음 유사 매칭)
            matched_model = model_prompt_map.get(user_prompt, None)
            
            if not matched_model and model_prompt_map:
                # 정확 매칭 실패시 유사 매칭 시도
                best_match = None
                best_ratio = 0
                
                for mapped_prompt, model in model_prompt_map.items():
                    # 간단한 유사도 계산 (공통 단어 비율)
                    user_words = set(user_prompt.lower().split())
                    mapped_words = set(mapped_prompt.lower().split())
                    
                    if user_words and mapped_words:
                        intersection = len(user_words & mapped_words)
                        union = len(user_words | mapped_words)
                        ratio = intersection / union
                        
                        if ratio > best_ratio and ratio > 0.6:  # 60% 이상 유사
                            best_ratio = ratio
                            best_match = model
                
                if best_match:
                    matched_model = best_match
                    print(f"Fuzzy matched '{user_prompt[:30]}...' to {best_match} (similarity: {best_ratio:.2f})")
            
            entry['source_model'] = matched_model or 'unknown'
        
        df_data = []
        # DataFrame 생성 - 각 모델별로 모든 엔트리를 중복 생성                                                                                                                                                                                                                                                                      │                                                                                                                                                                                                                                                                                                              │
        models_used = set()                                                                                                                                                                                                                                                                                                         │
        # 먼저 원본 LLM 결과에서 사용된 모델들 파악                                                                                                                                                                                                                                                                                 │
        if llm_results:
            models_used = set(llm_results.keys())  
        # 각 정규화된 엔트리에 대해 올바른 모델 정보 할당
        for model in models_used:
            category, parameters = self.extract_model_info(model)
            for entry in normalized_data:
                row = {
                    'category': category,
                    'llm_model': model,  # 해당 user_prompt를 생성한 원본 모델명 사용
                    'parameters': parameters,
                    'size': 'unknown',
                    'GPU_usage': 'unknown',
                    'object_name': entry['object_name'],
                    'seed': '',
                    'params': '',
                    'matched_image': '',
                    'object_name_clean': entry['object_name'],
                    'user_prompt': entry['user_prompt'],
                    'text_prompt': entry['text_prompt'],
                    'sha256': entry.get('sha256', ''),
                    'file_identifier': entry.get('file_identifier', '')
                }
                df_data.append(row)
        
        # 만약 모델이 없다면 unknown으로 처리 (이 부분은 위 로직에서 이미 처리되므로 사실상 불필요)
        if not models_used:
            for entry in normalized_data:
                row = {
                    'category': 'unknown',
                    'llm_model': 'unknown',
                    'parameters': 'unknown',
                    'size': 'unknown',
                    'GPU_usage': 'unknown',
                    'object_name': entry['object_name'],
                    'seed': '',
                    'params': '',
                    'matched_image': '',
                    'object_name_clean': entry['object_name'],
                    'user_prompt': entry['user_prompt'],
                    'text_prompt': entry['text_prompt'],
                    'sha256': entry.get('sha256', ''),
                    'file_identifier': entry.get('file_identifier', '')
                }
                df_data.append(row)
        
        # Excel 파일 저장
        df = pd.DataFrame(df_data)
        df.to_excel(output_path, index=False)
        
        print(f"✓ Unified Excel file created: {output_path}")
        print(f"✓ Total entries: {len(df_data)}")
        
        # 모델 분포 표시
        model_dist = pd.Series([row['llm_model'] for row in df_data]).value_counts().to_dict()
        print(f"✓ Original generation model distribution:")
        for model, count in model_dist.items():
            print(f"  - {model}: {count} prompts")
        
        return output_path

    def extract_model_info(self, model_name: str) -> tuple:
        """모델명에서 카테고리와 파라미터 정보 추출"""
        if model_name == 'unknown' or not model_name:
            return 'unknown', 'unknown'
        
        try:
            # 예: "qwen3:14b-q8_0" -> category="qwen3", parameters="14b"
            if ':' in model_name:
                category = model_name.split(':', 1)[0]
                param_part = model_name.split(':', 1)[1]
                
                # 파라미터 크기 추출 (예: "14b-q8_0" -> "14b")
                import re
                param_match = re.search(r'(\d+(?:\.\d+)?b)', param_part, re.IGNORECASE)
                parameters = param_match.group(1) if param_match else param_part.split('-')[0]
                
                return category, parameters
            else:
                return model_name, 'unknown'
                
        except Exception:
            return model_name, 'unknown'

    def extract_api_model_params(self, model_name: str) -> str:
        """외부 API 모델명에서 파라미터 정보 추출"""
        if not model_name:
            return 'unknown'
        
        try:
            # 다양한 API 모델명 패턴 처리
            model_lower = model_name.lower()
            
            # Groq/Llama 모델: llama-3.1-70b-versatile -> 70b
            if 'llama' in model_lower:
                import re
                match = re.search(r'(\d+(?:\.\d+)?b)', model_name, re.IGNORECASE)
                return match.group(1) if match else 'unknown'
            
            # OpenAI 모델: gpt-4o-mini -> 4o-mini
            elif 'gpt' in model_lower:
                if 'gpt-4o-mini' in model_lower:
                    return '4o-mini'
                elif 'gpt-4o' in model_lower:
                    return '4o'
                elif 'gpt-4' in model_lower:
                    return '4'
                elif 'gpt-3.5' in model_lower:
                    return '3.5'
                else:
                    return 'unknown'
            
            # Claude 모델: claude-3-5-haiku-20241022 -> 3.5-haiku  
            elif 'claude' in model_lower:
                if 'claude-3-5-sonnet' in model_lower:
                    return '3.5-sonnet'
                elif 'claude-3-5-haiku' in model_lower:
                    return '3.5-haiku'
                elif 'claude-3-opus' in model_lower:
                    return '3-opus'
                elif 'claude-3' in model_lower:
                    return '3'
                else:
                    return 'unknown'
            
            # Gemini 모델: gemini-1.5-flash -> 1.5-flash
            elif 'gemini' in model_lower:
                if 'gemini-1.5-pro' in model_lower:
                    return '1.5-pro'
                elif 'gemini-1.5-flash' in model_lower:
                    return '1.5-flash'
                elif 'gemini-1.5' in model_lower:
                    return '1.5'
                else:
                    return 'unknown'
            
            else:
                return 'unknown'
                
        except Exception:
            return 'unknown'

def main():
    """테스트 실행"""
    normalizer = EnhancedNormalizer()
    
    # 테스트 데이터
    test_output = '''
1. "Dark wooden table":"A dark wooden side table with carved legs and smooth surface"
2. "Red sports car":"A sleek red sports car with aerodynamic design and chrome details"
    '''
    
    # 임시 파일로 테스트
    test_file = "test_unified_output.txt"
    with open(test_file, 'w') as f:
        f.write(test_output)
    
    llm_results = {"test_model": test_file}
    result = normalizer.normalize_with_best_model(llm_results)
    
    print(f"Test result: {result}")
    
    # 정리
    if os.path.exists(test_file):
        os.remove(test_file)

if __name__ == "__main__":
    main()