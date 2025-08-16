#!/usr/bin/env python3
"""
엑셀 파일 샘플 수 체크 스크립트
사용자 지정 경로에서 part 폴더들을 자동으로 찾아 엑셀 파일들의 모델 수와 샘플 수를 체크
"""

import pandas as pd
import os
import glob
import sys
from pathlib import Path

def check_excel_counts(base_path):
    """
    지정된 경로에서 part 폴더들을 찾아 엑셀 파일들의 샘플 수를 체크
    
    Args:
        base_path: 기본 경로 (예: /path/to/sampled_data_100_random)
    """
    
    # 기본 경로 처리 - 파일이 아닌 prefix로 처리
    parent_dir = os.path.dirname(base_path)
    base_name = os.path.basename(base_path)
    
    # parent_dir가 존재하는지 확인
    if not os.path.exists(parent_dir):
        print(f"❌ 상위 디렉토리가 존재하지 않습니다: {parent_dir}")
        return
    
    # part 폴더 패턴으로 검색
    part_pattern = os.path.join(parent_dir, f"{base_name}_part*")
    part_folders = glob.glob(part_pattern)
    part_folders = [f for f in part_folders if os.path.isdir(f)]
    part_folders.sort()
    
    if not part_folders:
        print(f"❌ part 폴더를 찾을 수 없습니다: {part_pattern}")
        return
    
    print(f"🔍 찾은 part 폴더들:")
    for folder in part_folders:
        print(f"  - {os.path.basename(folder)}")
    print()
    
    total_results = []
    
    for part_folder in part_folders:
        part_name = os.path.basename(part_folder)
        print(f"📂 {part_name} 검사 중...")
        
        # 해당 폴더의 모든 .xlsx 파일 찾기
        excel_files = glob.glob(os.path.join(part_folder, "*.xlsx"))
        excel_files.sort()
        
        if not excel_files:
            print(f"  ⚠️  엑셀 파일이 없습니다.")
            continue
        
        for excel_file in excel_files:
            filename = os.path.basename(excel_file)
            
            try:
                # 엑셀 파일 읽기
                xl_file = pd.ExcelFile(excel_file)
                
                total_rows = 0
                model_count = 0
                sheet_info = []
                
                for sheet_name in xl_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    rows = len(df)
                    total_rows += rows
                    
                    # 모델 수 계산 (unique model 컬럼이 있는 경우)
                    if 'model' in df.columns:
                        unique_models = df['model'].nunique()
                        model_count = max(model_count, unique_models)
                        sheet_info.append(f"{sheet_name}({rows}행, {unique_models}모델)")
                    else:
                        sheet_info.append(f"{sheet_name}({rows}행)")
                
                # 결과 저장
                result = {
                    'part': part_name,
                    'file': filename,
                    'models': model_count if model_count > 0 else "N/A",
                    'samples': total_rows,
                    'sheets': sheet_info
                }
                total_results.append(result)
                
                # 출력
                print(f"  📄 {filename} : {model_count if model_count > 0 else 'N/A'}모델 : {total_rows}샘플")
                for sheet in sheet_info:
                    print(f"    └─ {sheet}")
                    
            except Exception as e:
                print(f"  ❌ {filename}: 오류 - {e}")
        
        print()
    
    # 전체 요약
    print("=" * 80)
    print("📊 전체 요약")
    print("=" * 80)
    
    if total_results:
        for result in total_results:
            print(f"{result['file']} : {result['models']}모델 : {result['samples']}샘플 ({result['part']})")
        
        print()
        print("📈 통계:")
        total_samples = sum(r['samples'] for r in total_results)
        total_files = len(total_results)
        print(f"- 총 파일 수: {total_files}")
        print(f"- 총 샘플 수: {total_samples}")
        
        # 파일명별 그룹화 통계
        file_stats = {}
        for result in total_results:
            filename = result['file']
            if filename not in file_stats:
                file_stats[filename] = {'count': 0, 'total_samples': 0, 'parts': []}
            file_stats[filename]['count'] += 1
            file_stats[filename]['total_samples'] += result['samples']
            file_stats[filename]['parts'].append(result['part'])
        
        print(f"\n📋 파일명별 통계:")
        for filename, stats in file_stats.items():
            print(f"- {filename}: {stats['count']}개 part, 총 {stats['total_samples']}샘플")
            print(f"  └─ 등장 part: {', '.join(stats['parts'])}")
    
    else:
        print("❌ 분석할 데이터가 없습니다.")

def main():
    if len(sys.argv) != 2:
        print("사용법: python check_excel_counts.py <기본경로>")
        print("예시: python check_excel_counts.py /home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs/sampled_data_100_random")
        sys.exit(1)
    
    base_path = sys.argv[1]
    check_excel_counts(base_path)

if __name__ == "__main__":
    main()