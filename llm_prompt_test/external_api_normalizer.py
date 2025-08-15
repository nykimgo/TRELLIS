#!/usr/bin/env python3
"""
외부 API를 사용한 고속 정규화 도구
OpenAI, Claude, Gemini, Groq 등 외부 API 지원
"""

import os
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
import requests
import re

class ExternalAPINormalizer:
    def __init__(self, config_file: str = "api_config.json"):
        """
        Args:
            config_file: API 설정 파일 경로
        """
        self.config = self.load_config(config_file)
        self.unified_prompt = """
You are an expert text processing assistant. Your task is to process LLM-generated prompt enhancement outputs and extract structured information.

Given raw LLM output containing original prompts and their enhanced versions, extract and return the data in this exact JSON format for each pair:

{"object_name": "<main_object>", "text_prompt": "<enhanced_detailed_prompt>", "llm_model": "<llm_model_name>"}

Rules:
1. object_name: Extract the main object from the enhanced prompt (1-2 words, singular form, no articles) 
2. text_prompt: Enhanced detailed prompt (clean, descriptive)
3. llm_model: The name of the LLM model that generated this specific prompt (extract from section headers like "=== Output from gemma3:1b ===")
4. Return ONLY valid JSON objects, one per line
5. No explanations, headers, or additional text

Examples:
Input: === Output from gemma3:1b ===
1. "Dark wooden table":"A dark wooden side table with carved legs..."
Output: {"object_name": "Table", "text_prompt": "A dark wooden side table with carved legs and a smooth surface finish", "llm_model": "gemma3:1b"}

Now process this input:

"""

    def load_config(self, config_file: str) -> Dict:
        """API 설정 로드"""
        default_config = {
            "preferred_apis": ["groq", "openai", "claude", "gemini"],
            "groq": {
                "api_key": "",
                "base_url": "https://api.groq.com/openai/v1",
                "model": "llama-3.1-70b-versatile",
                "enabled": False
            },
            "openai": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model": "gpt-4o-mini",
                "enabled": False
            },
            "claude": {
                "api_key": "",
                "base_url": "https://api.anthropic.com",
                "model": "claude-3-5-haiku-20241022",
                "enabled": False
            },
            "gemini": {
                "api_key": "",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "model": "gemini-1.5-flash",
                "enabled": False
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}")
        else:
            # 기본 설정 파일 생성
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
            print(f"Created config file: {config_file}")
            print("Please add your API keys to the config file")
        
        return default_config

    def call_groq_api(self, prompt: str) -> Optional[str]:
        """Groq API 호출"""
        config = self.config["groq"]
        if not config["enabled"] or not config["api_key"]:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 4000
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"Groq API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Groq API call failed: {e}")
            return None

    def call_openai_api(self, prompt: str) -> Optional[str]:
        """OpenAI API 호출"""
        config = self.config["openai"]
        if not config["enabled"] or not config["api_key"]:
            return None
            
        try:
            headers = {
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 4000
            }
            
            response = requests.post(
                f"{config['base_url']}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                print(f"OpenAI API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"OpenAI API call failed: {e}")
            return None

    def call_claude_api(self, prompt: str) -> Optional[str]:
        """Claude API 호출"""
        config = self.config["claude"]
        if not config["enabled"] or not config["api_key"]:
            return None
            
        try:
            headers = {
                "x-api-key": config['api_key'],
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": config["model"],
                "max_tokens": 4000,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            response = requests.post(
                f"{config['base_url']}/v1/messages",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["content"][0]["text"]
            else:
                print(f"Claude API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Claude API call failed: {e}")
            return None

    def call_gemini_api(self, prompt: str) -> Optional[str]:
        """Gemini API 호출"""
        config = self.config["gemini"]
        if not config["enabled"] or not config["api_key"]:
            return None
            
        try:
            url = f"{config['base_url']}/models/{config['model']}:generateContent?key={config['api_key']}"
            
            data = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 4000
                }
            }
            
            response = requests.post(url, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                print(f"Gemini API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"Gemini API call failed: {e}")
            return None

    def normalize_with_external_api(self, llm_results: Dict[str, str]) -> List[Dict]:
        """외부 API를 사용한 고속 정규화"""
        
        # 출력 디렉토리 설정
        output_dir = Path(list(llm_results.values())[0]).parent
        
        # 모든 LLM 출력을 하나로 합치기
        combined_content = ""
        for model, file_path in llm_results.items():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                combined_content += f"\n=== Output from {model} ===\n{content}\n"
        
        # 통합 프롬프트 생성
        full_prompt = self.unified_prompt + combined_content
        
        print("🚀 Using external API for fast normalization...")
        
        # API 우선순위에 따라 시도
        for api_name in self.config["preferred_apis"]:
            if not self.config[api_name]["enabled"]:
                continue
                
            print(f"Trying {api_name.upper()} API...")
            start_time = time.time()
            
            # 전송할 프롬프트 저장
            prompt_file = output_dir / f"prompt_sent_to_{api_name}.txt"
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(f"=== Prompt sent to {api_name.upper()} API ===\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model: {self.config[api_name]['model']}\n")
                f.write("=" * 60 + "\n\n")
                f.write(full_prompt)
            print(f"📝 Saved prompt to: {prompt_file}")
            
            # API 호출
            if api_name == "groq":
                result = self.call_groq_api(full_prompt)
            elif api_name == "openai":
                result = self.call_openai_api(full_prompt)
            elif api_name == "claude":
                result = self.call_claude_api(full_prompt)
            elif api_name == "gemini":
                result = self.call_gemini_api(full_prompt)
            else:
                continue
            
            if result:
                elapsed = time.time() - start_time
                print(f"✅ {api_name.upper()} API successful! ({elapsed:.1f}s)")
                
                # 원본 응답 저장 (가공되지 않은 상태)
                raw_response_file = output_dir / f"raw_response_from_{api_name}.txt"
                with open(raw_response_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== Raw response from {api_name.upper()} API ===\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Model: {self.config[api_name]['model']}\n")
                    f.write(f"Response time: {elapsed:.1f}s\n")
                    f.write("=" * 60 + "\n\n")
                    f.write(result)
                print(f"📥 Saved raw response to: {raw_response_file}")
                
                # 정규화된 결과 저장 (기존 코드 유지)
                normalized_file = output_dir / f"external_api_normalized_{api_name}.txt"
                with open(normalized_file, 'w', encoding='utf-8') as f:
                    f.write(result)
                print(f"📋 Saved normalized result to: {normalized_file}")
                
                # JSON 파싱 및 모델 정보 추가
                normalized_data = self.parse_json_output(result)
                
                if normalized_data:
                    # 파싱된 JSON 데이터도 별도 저장
                    parsed_json_file = output_dir / f"parsed_json_from_{api_name}.json"
                    with open(parsed_json_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            "metadata": {
                                "api_used": api_name,
                                "model": self.config[api_name]['model'],
                                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                                "response_time": f"{elapsed:.1f}s",
                                "total_entries": len(normalized_data)
                            },
                            "normalized_data": normalized_data
                        }, f, indent=2, ensure_ascii=False)
                    print(f"🔍 Saved parsed JSON to: {parsed_json_file}")
                    
                    # 외부 API 사용시 통합된 모델 정보 추가
                    for entry in normalized_data:
                        if 'api_used' not in entry:
                            entry['api_used'] = api_name
                    
                    print(f"✅ Successfully parsed {len(normalized_data)} entries")
                    return normalized_data
                else:
                    print(f"⚠️ {api_name.upper()} returned data but JSON parsing failed")
                    # 파싱 실패한 경우에도 실패 파일 저장
                    failed_file = output_dir / f"parsing_failed_{api_name}.txt"
                    with open(failed_file, 'w', encoding='utf-8') as f:
                        f.write(f"=== JSON Parsing Failed for {api_name.upper()} ===\n")
                        f.write(f"Original response:\n{result}")
                    print(f"❌ Saved failed parsing attempt to: {failed_file}")
            else:
                print(f"❌ {api_name.upper()} API failed")
                # API 호출 실패 로그 저장
                failed_file = output_dir / f"api_call_failed_{api_name}.txt"
                with open(failed_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== API Call Failed for {api_name.upper()} ===\n")
                    f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Model: {self.config[api_name]['model']}\n")
                    f.write("Prompt that was attempted to be sent:\n")
                    f.write("=" * 60 + "\n")
                    f.write(full_prompt)
        
        print("❌ All external APIs failed, falling back to local method")
        return self.fallback_normalize(llm_results)

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
                    if all(key in data for key in ['object_name', 'text_prompt']):
                        entry = {
                            'object_name': str(data['object_name']).strip(),
                            'text_prompt': str(data['text_prompt']).strip()
                        }
                        
                        # llm_model 필드가 있으면 추가 (선택적)
                        if 'llm_model' in data:
                            entry['llm_model'] = str(data['llm_model']).strip()
                        
                        normalized_data.append(entry)
                        
                except json.JSONDecodeError:
                    continue
        
        return normalized_data

    def fallback_normalize(self, llm_results: Dict[str, str]) -> List[Dict]:
        """Fallback to rule-based normalization"""
        print("Using fallback rule-based normalization")
        
        try:
            from enhanced_normalizer import EnhancedNormalizer
            normalizer = EnhancedNormalizer()
            return normalizer.fallback_normalize(llm_results)
        except:
            # 최소한의 fallback
            return []

    def setup_wizard(self):
        """API 설정 마법사"""
        print("🔧 External API Setup Wizard")
        print("=" * 40)
        
        apis = ["groq", "openai", "claude", "gemini"]
        
        for api in apis:
            print(f"\n📡 {api.upper()} API Setup:")
            
            if api == "groq":
                print("  - Very fast inference (recommended)")
                print("  - Free tier available")
                print("  - Get API key from: https://console.groq.com")
            elif api == "openai":
                print("  - Most reliable")
                print("  - Paid service")
                print("  - Get API key from: https://platform.openai.com")
            elif api == "claude":
                print("  - High quality")
                print("  - Paid service")
                print("  - Get API key from: https://console.anthropic.com")
            elif api == "gemini":
                print("  - Good performance")
                print("  - Free tier available")
                print("  - Get API key from: https://makersuite.google.com")
            
            enable = input(f"Enable {api.upper()}? (y/n): ").lower().strip() == 'y'
            
            if enable:
                api_key = input(f"Enter {api.upper()} API key: ").strip()
                self.config[api]["api_key"] = api_key
                self.config[api]["enabled"] = True
                print(f"✅ {api.upper()} configured!")
            else:
                self.config[api]["enabled"] = False
        
        # 설정 저장
        with open("api_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print("\n🎉 Setup complete! Configuration saved to api_config.json")

def main():
    """테스트 및 설정"""
    normalizer = ExternalAPINormalizer()
    
    # 설정되지 않은 API가 있으면 설정 마법사 실행
    has_enabled_api = any(api["enabled"] for api in [
        normalizer.config["groq"], 
        normalizer.config["openai"], 
        normalizer.config["claude"], 
        normalizer.config["gemini"]
    ])
    
    if not has_enabled_api:
        print("No APIs configured. Running setup wizard...")
        normalizer.setup_wizard()
    else:
        print("External API normalizer is ready!")
        enabled_apis = [
            name for name in ["groq", "openai", "claude", "gemini"] 
            if normalizer.config[name]["enabled"]
        ]
        print(f"Enabled APIs: {', '.join(enabled_apis)}")

if __name__ == "__main__":
    main()