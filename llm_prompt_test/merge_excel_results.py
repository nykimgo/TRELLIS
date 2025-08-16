#!/usr/bin/env python3
"""
엑셀 파일 병합 스크립트
각 part 폴더의 엑셀 파일들을 하나로 합침
"""

import pandas as pd
import os
import glob
from pathlib import Path

def merge_excel_files():
    # 기본 경로 설정
    base_path = "/home/sr/TRELLIS/llm_prompt_test/prompt_generation_outputs"
    
    # part 폴더 목록 (part01~part05)
    part_folders = [
        "sampled_data_100_random_part01",
        "sampled_data_100_random_part02",
        "sampled_data_100_random_part03", 
        "sampled_data_100_random_part04",
        "sampled_data_100_random_part05"
    ]
    
    # 테스트용: part01, part02만 사용
    test_mode = False  # 전체 part 처리
    if test_mode:
        part_folders = part_folders[:2]  # part01, part02만
        print("테스트 모드: part01, part02만 처리")
    else:
        print("전체 모드: part01~part05 처리")
    
    all_data = []
    
    for part_folder in part_folders:
        folder_path = os.path.join(base_path, part_folder)
        
        # 폴더 존재 확인
        if not os.path.exists(folder_path):
            print(f"폴더가 존재하지 않음: {folder_path}")
            continue
            
        print(f"처리 중: {part_folder}")
        
        # 해당 폴더의 모든 .xlsx 파일 찾기
        excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        
        for excel_file in excel_files:
            print(f"  파일 읽는 중: {os.path.basename(excel_file)}")
            
            try:
                # 엑셀 파일 읽기 (모든 시트)
                xl_file = pd.ExcelFile(excel_file)
                
                for sheet_name in xl_file.sheet_names:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    
                    # 메타데이터 추가
                    df['source_part'] = part_folder
                    df['source_file'] = os.path.basename(excel_file)
                    df['source_sheet'] = sheet_name
                    
                    all_data.append(df)
                    print(f"    시트 '{sheet_name}': {len(df)}행 추가")
                    
            except Exception as e:
                print(f"  오류 발생 - {excel_file}: {e}")
                continue
    
    if not all_data:
        print("병합할 데이터가 없습니다.")
        return
    
    # 모든 데이터 병합
    print("\n데이터 병합 중...")
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # 출력 파일명 생성
    output_file = os.path.join(base_path, "sampled_data_100_random_results.xlsx")
    
    # 엑셀 파일로 저장
    print(f"저장 중: {output_file}")
    merged_df.to_excel(output_file, index=False)
    
    print(f"\n완료!")
    print(f"총 {len(merged_df)}행의 데이터가 병합되었습니다.")
    print(f"출력 파일: {output_file}")
    
    # 요약 정보 출력
    print(f"\n요약:")
    print(f"- 처리된 part 수: {len(part_folders)}")
    print(f"- 총 행 수: {len(merged_df)}")
    print(f"- source_part 분포:")
    print(merged_df['source_part'].value_counts())

if __name__ == "__main__":
    merge_excel_files()