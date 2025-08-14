#!/usr/bin/env python3
"""
객체명 자동 생성 도구
프롬프트에서 대표 객체명을 추출하는 LLM 기반 도구
"""

import subprocess
import json
import re
from pathlib import Path
from typing import List, Dict

class ObjectNameGenerator:
    def __init__(self):
        self.object_name_prompt = """
You are an expert at identifying the main object in descriptive text prompts.

Given a text prompt, identify and return ONLY the main object name in a single word or short phrase (1-3 words maximum).

Rules:
1. Return only the primary object being described
2. Use singular form (e.g., "chair" not "chairs") 
3. Use simple, common English words
4. No articles (a, an, the)
5. No adjectives or descriptors
6. If multiple objects, choose the most prominent one

Examples:
Input: "Dark wooden side table with flat top and drawer"
Output: Table

Input: "Modern white armchair with curved backrest"
Output: Armchair

Input: "Bushy lavender plant with purple flowers"
Output: Plant

Input: "Rectangular kitchen island with storage"
Output: Island

Now process this prompt:
"""

    def generate_object_name(self, prompt: str, model: str = "qwen3:1.7b") -> str:
        """단일 프롬프트에서 객체명 생성"""
        full_prompt = self.object_name_prompt + prompt
        
        try:
            # Ollama 실행
            result = subprocess.run(
                ["ollama", "run", model],
                input=full_prompt,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # 출력에서 객체명 추출
                output = result.stdout.strip()
                
                # 첫 번째 줄만 가져오고 정리
                object_name = output.split('\n')[0].strip()
                
                # 불필요한 문자 제거
                object_name = re.sub(r'[^\w\s]', '', object_name)
                object_name = ' '.join(object_name.split()[:3])  # 최대 3단어
                
                return object_name.title() if object_name else "Unknown"
            else:
                print(f"LLM error: {result.stderr}")
                return self._fallback_object_extraction(prompt)
                
        except Exception as e:
            print(f"Error generating object name: {e}")
            return self._fallback_object_extraction(prompt)
    
    def _fallback_object_extraction(self, prompt: str) -> str:
        """LLM 실패시 rule-based 객체명 추출"""
        words = prompt.lower().split()
        
        # 일반적인 3D 객체 키워드들
        furniture_keywords = [
            'table', 'chair', 'bed', 'sofa', 'couch', 'desk', 'cabinet', 
            'shelf', 'drawer', 'wardrobe', 'dresser', 'nightstand', 'bench',
            'stool', 'armchair', 'bookshelf', 'sideboard', 'island'
        ]
        
        lighting_keywords = [
            'lamp', 'light', 'chandelier', 'lantern', 'candle', 'bulb', 'fixture'
        ]
        
        appliance_keywords = [
            'television', 'tv', 'computer', 'monitor', 'phone', 'telephone',
            'radio', 'speaker', 'microwave', 'oven', 'refrigerator'
        ]
        
        decorative_keywords = [
            'vase', 'bowl', 'cup', 'plate', 'bottle', 'pot', 'sculpture',
            'painting', 'picture', 'mirror', 'clock', 'book'
        ]
        
        plant_keywords = [
            'plant', 'flower', 'tree', 'bush', 'lavender', 'rose', 'cactus'
        ]
        
        building_keywords = [
            'house', 'building', 'castle', 'cottage', 'tower', 'shed', 'barn'
        ]
        
        vehicle_keywords = [
            'car', 'truck', 'bicycle', 'motorcycle', 'boat', 'ship', 'plane',
            'helicopter', 'train', 'bus'
        ]
        
        all_keywords = {
            **{k: k.title() for k in furniture_keywords},
            **{k: k.title() for k in lighting_keywords},
            **{k: k.title() for k in appliance_keywords},
            **{k: k.title() for k in decorative_keywords},
            **{k: k.title() for k in plant_keywords},
            **{k: k.title() for k in building_keywords},
            **{k: k.title() for k in vehicle_keywords}
        }
        
        # 키워드 매칭
        for word in words:
            if word in all_keywords:
                return all_keywords[word]
        
        # 특수 패턴 매칭
        if any(w in words for w in ['rotary', 'telephone', 'phone']):
            return "Telephone"
        if any(w in words for w in ['orb', 'sphere', 'ball']):
            return "Orb"
        if any(w in words for w in ['pedestal', 'stand', 'base']):
            return "Pedestal"
        
        # 첫 번째 명사로 추정되는 단어 반환
        for word in words:
            if len(word) > 2 and word.isalpha():
                return word.title()
        
        return "Object"
    
    def batch_generate_object_names(self, prompts: List[str], model: str = "qwen3:1.7b") -> List[str]:
        """여러 프롬프트에 대한 객체명 배치 생성"""
        print(f"Generating object names for {len(prompts)} prompts using {model}")
        
        object_names = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"Processing {i}/{len(prompts)}: {prompt[:50]}...")
            object_name = self.generate_object_name(prompt, model)
            object_names.append(object_name)
            print(f"→ {object_name}")
        
        return object_names
    
    def update_excel_with_object_names(self, excel_file: str, model: str = "qwen3:1.7b") -> str:
        """Excel 파일의 object_name 컬럼을 자동으로 업데이트"""
        try:
            import pandas as pd
            
            # Excel 파일 읽기
            df = pd.read_excel(excel_file)
            
            if 'user_prompt' not in df.columns:
                print("Error: 'user_prompt' column not found in Excel file")
                return ""
            
            # 객체명 생성
            prompts = df['user_prompt'].tolist()
            object_names = self.batch_generate_object_names(prompts, model)
            
            # DataFrame 업데이트
            df['object_name'] = object_names  # 동일한 값으로 설정
            
            # 업데이트된 파일 저장
            updated_file = excel_file.replace('.xlsx', '_with_object_names.xlsx')
            df.to_excel(updated_file, index=False)
            
            print(f"✓ Updated Excel file saved to: {updated_file}")
            return updated_file
            
        except ImportError:
            print("Error: pandas not installed. Install with: pip install pandas openpyxl")
            return ""
        except Exception as e:
            print(f"Error updating Excel file: {e}")
            return ""

def main():
    import sys
    
    generator = ObjectNameGenerator()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python object_name_generator.py <excel_file> [model_name]")
        print("  python object_name_generator.py \"single prompt text\" [model_name]")
        return
    
    input_arg = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3:1.7b"
    
    if input_arg.endswith('.xlsx'):
        # Excel 파일 처리
        result = generator.update_excel_with_object_names(input_arg, model)
        if result:
            print(f"Success! Updated file: {result}")
    else:
        # 단일 프롬프트 처리
        object_name = generator.generate_object_name(input_arg, model)
        print(f"Object name: {object_name}")

if __name__ == "__main__":
    main()