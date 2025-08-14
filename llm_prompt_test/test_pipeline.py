#!/usr/bin/env python3
"""
파이프라인 테스트 스크립트
소규모 데이터로 파이프라인 기능을 테스트
"""

import sys
import os
from pathlib import Path

def test_basic_functionality(metadata_path):
    """기본 기능 테스트"""
    print("=== Testing Basic Functionality ===")
    
    # 1. 파일 존재 확인
    required_files = [
        'automated_prompt_generator.py',
        'llm_output_normalizer.py', 
        'object_name_generator.py',
        'run_automated_pipeline.py',
        'pipeline_config.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✓ {file} exists")
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    
    # 2. 모듈 임포트 테스트
    try:
        from automated_prompt_generator import PromptGenerator
        print("✓ PromptGenerator import successful")
        
        from llm_output_normalizer import LLMOutputNormalizer
        print("✓ LLMOutputNormalizer import successful")
        
        from object_name_generator import ObjectNameGenerator
        print("✓ ObjectNameGenerator import successful")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # 3. 메타데이터 파일 확인
    if os.path.exists(metadata_path):
        print(f"✓ Metadata file exists: {metadata_path}")
    else:
        print(f"✗ Metadata file not found: {metadata_path}")
        print("Available datasets:")
        datasets_dir = "../datasets"
        if os.path.exists(datasets_dir):
            for item in os.listdir(datasets_dir):
                item_path = os.path.join(datasets_dir, item)
                if os.path.isdir(item_path):
                    metadata_file = os.path.join(item_path, "metadata.csv")
                    if os.path.exists(metadata_file):
                        print(f"  - {item}: {metadata_file}")
        return False
    
    print("✓ All basic tests passed!")
    return True

def test_small_sample(metadata_path):
    """소규모 샘플로 파이프라인 테스트"""
    print("\n=== Testing Small Sample Pipeline ===")
    
    try:
        from automated_prompt_generator import PromptGenerator
        
        # 매우 작은 샘플로 테스트 (5개)
        generator = PromptGenerator(metadata_path, num_samples=5)
        
        # 데이터 로드 테스트
        data = generator.load_and_sample_data()
        if data:
            print(f"✓ Successfully loaded {len(data)} samples")
            
            # 첫 번째 샘플 출력
            if data:
                print(f"Sample caption: {data[0]['original_caption']}")
        else:
            print("✗ Failed to load sample data")
            return False
        
        # prompts.txt 생성 테스트
        prompts_file = generator.generate_prompts_file("test_prompts.txt")
        if os.path.exists(prompts_file):
            print(f"✓ Generated prompts file: {prompts_file}")
        else:
            print("✗ Failed to generate prompts file")
            return False
        
        print("✓ Small sample test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Small sample test failed: {e}")
        return False

def test_object_name_extraction():
    """객체명 추출 테스트"""
    print("\n=== Testing Object Name Extraction ===")
    
    try:
        from object_name_generator import ObjectNameGenerator
        
        generator = ObjectNameGenerator()
        
        # 테스트 프롬프트들
        test_prompts = [
            "Dark wooden side table with flat top and drawer",
            "Modern white armchair with curved backrest", 
            "Bushy lavender plant with purple flowers",
            "Rectangular kitchen island with storage",
            "Vintage copper rotary telephone"
        ]
        
        print("Testing fallback object extraction (without LLM):")
        for prompt in test_prompts:
            object_name = generator._fallback_object_extraction(prompt)
            print(f"  '{prompt[:30]}...' → {object_name}")
        
        print("✓ Object name extraction test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Object name extraction test failed: {e}")
        return False

def test_enhanced_normalization():
    """향상된 정규화 기능 테스트"""
    print("\n=== Testing Enhanced Normalization ===")
    
    try:
        from enhanced_normalizer import EnhancedNormalizer
        
        normalizer = EnhancedNormalizer()
        
        # 테스트 LLM 출력 생성
        test_output = '''
Here are the rewritten prompts:

1. "Vintage copper rotary telephone with intricate detailing."
:"An antique, intricately detailed copper rotary phone with ornate engravings, rounded body, and a circular dial pad, sitting on a worn wooden surface."

2. "Two-story brick house with red roof and fence."
:"A traditional two-story brick house with a bright red terra cotta tiled roof, white trim, green grass, and a white picket fence surrounding the property."
        '''
        
        # 임시 파일로 저장
        test_file = "test_enhanced_output.txt"
        with open(test_file, 'w') as f:
            f.write(test_output)
        
        # 향상된 정규화 테스트 (fallback 방식으로)
        llm_results = {"test_model": test_file}
        normalized_data = normalizer.fallback_normalize(llm_results)
        
        if normalized_data:
            print(f"✓ Enhanced normalization successful: {len(normalized_data)} entries")
            
            # 결과 확인
            for i, entry in enumerate(normalized_data[:2]):
                print(f"Entry {i+1}:")
                print(f"  Object: {entry['object_name']}")
                print(f"  User prompt: {entry['user_prompt']}")
                print(f"  Text prompt: {entry['text_prompt'][:50]}...")
            
            # 정리
            os.remove(test_file)
        else:
            print("✗ Enhanced normalization failed")
            return False
        
        print("✓ Enhanced normalization test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced normalization test failed: {e}")
        return False

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
    """테스트 실행"""
    print("Running Pipeline Tests...\n")
    
    # 명령행 인자로 메타데이터 경로가 주어졌는지 확인
    if len(sys.argv) > 1:
        metadata_path = sys.argv[1]
        if not os.path.exists(metadata_path):
            print(f"Error: Metadata file not found: {metadata_path}")
            sys.exit(1)
    else:
        # 사용자 인터랙티브 선택
        metadata_path = get_metadata_path()
        if not metadata_path:
            print("No metadata file selected. Exiting.")
            sys.exit(1)
    
    print(f"Using metadata file: {metadata_path}\n")
    
    tests = [
        ("Basic Functionality", lambda: test_basic_functionality(metadata_path)),
        ("Small Sample", lambda: test_small_sample(metadata_path)),
        ("Object Name Extraction", test_object_name_extraction),
        ("Enhanced Normalization", test_enhanced_normalization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Install required Ollama models:")
        print("   ollama pull gemma3:1b")
        print("   ollama pull qwen3:1.7b")
        print("2. Run the full pipeline:")
        print(f"   python run_automated_pipeline.py {metadata_path}")
    else:
        print("❌ Some tests failed. Please fix the issues before running the pipeline.")
        sys.exit(1)

if __name__ == "__main__":
    main()