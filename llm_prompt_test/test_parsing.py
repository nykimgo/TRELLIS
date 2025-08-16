#\!/usr/bin/env python3

from automated_prompt_generator import PromptGenerator

# CSV 파일로 테스트  
generator = PromptGenerator(
    metadata_path='../datasets/HSSD/metadata.csv',
    num_samples=20,
    csv_file='/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part01.csv'
)

# CSV 데이터 로드
generator.selected_data = generator.load_from_csv('/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random_part01.csv')
print(f'Loaded {len(generator.selected_data)} samples from CSV')

# 파싱 테스트
pairs = generator.parse_llm_output('/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/detailed_prompt_deepseek-r1_14b-qwen-distill-q8_0.txt')
print(f'Parsed {len(pairs)} pairs')

if len(pairs) >= 5:
    print('First 5 pairs:')
    for i, (orig, enh) in enumerate(pairs[:5]):
        print(f'{i+1}. Original: {orig}')
        print(f'   Enhanced: {enh[:100]}...')
        print()
EOF < /dev/null
