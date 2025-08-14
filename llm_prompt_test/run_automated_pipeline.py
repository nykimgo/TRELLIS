#!/usr/bin/env python3
"""
통합 자동화 파이프라인 실행기
전체 프롬프트 증강 프로세스를 자동으로 실행
"""

import os
import sys
import subprocess
from pathlib import Path
import json

def check_dependencies():
    """필수 의존성 확인"""
    print("Checking dependencies...")
    
    # Ollama 설치 확인
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Ollama is installed")
        else:
            print("✗ Ollama not found. Please install Ollama first.")
            return False
    except FileNotFoundError:
        print("✗ Ollama not found. Please install Ollama first.")
        return False
    
    # Python 패키지 확인
    required_packages = ['pandas', 'openpyxl']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print(f"Please install missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def extract_model_params(model_name):
    """모델명에서 파라미터 크기 추출 (예: qwen3:32b-q8_0 -> 32)"""
    try:
        # ':' 이후 부분 추출
        if ':' in model_name:
            param_part = model_name.split(':', 1)[1]
            # 'b' 앞의 숫자 추출
            import re
            match = re.search(r'(\d+(?:\.\d+)?)b', param_part, re.IGNORECASE)
            if match:
                return float(match.group(1))
    except:
        pass
    return 0

def get_best_model_by_params(models):
    """파라미터 크기가 가장 큰 모델 선택"""
    if not models:
        return None
    
    best_model = max(models, key=extract_model_params)
    best_params = extract_model_params(best_model)
    
    print(f"Selected best model by parameter size: {best_model} ({best_params}B params)")
    return best_model

def check_ollama_models(models):
    """설치된 Ollama 모델 확인"""
    print("Checking available Ollama models...")
    
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            available_models = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if line.strip():
                    model_name = line.split()[0]
                    available_models.append(model_name)
            
            print(f"Available models: {available_models}")
            
            # 요청된 모델이 설치되어 있는지 확인
            valid_models = []
            for model in models:
                if model in available_models:
                    params = extract_model_params(model)
                    print(f"✓ {model} is available ({params}B params)")
                    valid_models.append(model)
                else:
                    print(f"✗ {model} is not installed")
                    print(f"  Install with: ollama pull {model}")
            
            return valid_models
        else:
            print("Error checking Ollama models")
            return []
            
    except Exception as e:
        print(f"Error checking models: {e}")
        return []

def run_pipeline(metadata_path, num_samples, models, normalize_output=True, generate_object_names=True):
    """전체 파이프라인 실행"""
    print("=== Starting Automated Prompt Augmentation Pipeline ===")
    print(f"Metadata: {metadata_path}")
    print(f"Samples: {num_samples}")
    print(f"Models: {models}")
    print(f"Normalize output: {normalize_output}")
    print(f"Generate object names: {generate_object_names}")
    print()
    
    # 1. 의존성 확인
    if not check_dependencies():
        return False
    
    # 2. Ollama 모델 확인
    valid_models = check_ollama_models(models)
    if not valid_models:
        print("No valid models found. Please install required models first.")
        return False
    
    print(f"Using models: {valid_models}")
    print()
    
    # 3. 메인 프롬프트 생성 실행
    print("Step 1: Running main prompt generation...")
    try:
        from automated_prompt_generator import PromptGenerator
        
        generator = PromptGenerator(metadata_path, num_samples)
        excel_path = generator.run_full_pipeline(valid_models)
        
        if not excel_path:
            print("Main pipeline failed!")
            return False
            
        print(f"✓ Main pipeline completed: {excel_path}")
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        return False
    
    # 4. 향상된 정규화는 메인 파이프라인에서 자동으로 처리됨
    print("\n✓ Enhanced normalization and object name generation completed in main pipeline")
    
    print(f"\n=== Pipeline Completed Successfully ===")
    print(f"Final result: {excel_path}")
    print(f"Output directory: {Path(excel_path).parent}")
    
    return True

def get_metadata_path():
    """사용자로부터 메타데이터 파일 경로 입력받기"""
    print("Available datasets:")
    datasets_dir = "../datasets"
    available_datasets = []
    
    if os.path.exists(datasets_dir):
        for item in os.listdir(datasets_dir):
            item_path = os.path.join(datasets_dir, item)
            if os.path.isdir(item_path):
                metadata_file = os.path.join(item_path, "metadata.csv")
                if os.path.exists(metadata_file):
                    available_datasets.append((item, metadata_file))
                    print(f"  {len(available_datasets)}. {item}: {metadata_file}")
    
    if not available_datasets:
        print("No datasets found with metadata.csv files!")
        return None
    
    print(f"  {len(available_datasets) + 1}. Custom path")
    
    while True:
        try:
            choice = input(f"\nSelect dataset (1-{len(available_datasets) + 1}): ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_datasets):
                    return available_datasets[choice_num - 1][1]
                elif choice_num == len(available_datasets) + 1:
                    custom_path = input("Enter custom metadata.csv path: ").strip()
                    if os.path.exists(custom_path):
                        return custom_path
                    else:
                        print(f"File not found: {custom_path}")
                        continue
            
            print("Invalid choice. Please try again.")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None
        except Exception as e:
            print(f"Error: {e}")

def main():
    """메인 함수"""
    # 기본 설정
    config = {
        'metadata_path': '../datasets/HSSD/metadata.csv',
        'num_samples': 100,
        'models': [
            'gemma3:1b',
            'qwen3:1.7b', 
            'qwen3:14b'
        ],
        'normalize_output': True,
        'generate_object_names': True
    }
    
    # 설정 파일이 있다면 로드
    config_file = 'pipeline_config.json'
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                config.update(user_config)
            print(f"Loaded configuration from {config_file}")
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    # 명령행 인자 처리
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python run_automated_pipeline.py [metadata_path] [num_samples]")
            print(f"Current config: {json.dumps(config, indent=2)}")
            print(f"\nTo customize, create {config_file} with your settings")
            return
        
        config['metadata_path'] = sys.argv[1]
    else:
        # 메타데이터 경로가 주어지지 않았거나 파일이 없으면 사용자 선택
        if not os.path.exists(config['metadata_path']):
            print(f"Default metadata file not found: {config['metadata_path']}")
            metadata_path = get_metadata_path()
            if not metadata_path:
                print("No metadata file selected. Exiting.")
                return
            config['metadata_path'] = metadata_path
    
    if len(sys.argv) > 2:
        config['num_samples'] = int(sys.argv[2])
    
    # 파이프라인 실행
    success = run_pipeline(
        config['metadata_path'],
        config['num_samples'], 
        config['models'],
        config['normalize_output'],
        config['generate_object_names']
    )
    
    if success:
        print("\n🎉 All done! Check the output directory for results.")
    else:
        print("\n❌ Pipeline failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()