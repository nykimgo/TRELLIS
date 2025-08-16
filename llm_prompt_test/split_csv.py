#!/usr/bin/env python3
import pandas as pd
import os
import sys
import math

def split_file(file_path, num_parts):
    """
    CSV 또는 Excel 파일을 지정된 개수로 균등하게 나누어 저장합니다.
    
    Args:
        file_path (str): 입력 파일 경로 (.csv 또는 .xlsx)
        num_parts (int): 나눌 파일 개수
    """
    # 파일 확장자 확인
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    # 파일 읽기
    if extension == '.csv':
        df = pd.read_csv(file_path)
    elif extension == '.xlsx':
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {extension}")
    
    total_rows = len(df)
    
    # 각 파트당 행 수 계산
    rows_per_part = math.ceil(total_rows / num_parts)
    
    # 파일명과 확장자 분리
    base_path = os.path.splitext(file_path)[0]
    
    print(f"총 {total_rows}개 행을 {num_parts}개 파일로 나눕니다.")
    print(f"각 파트당 최대 {rows_per_part}개 행")
    
    # 파일 나누어 저장
    for i in range(num_parts):
        start_idx = i * rows_per_part
        end_idx = min((i + 1) * rows_per_part, total_rows)
        
        # 범위 내에 데이터가 있을 때만 파일 생성
        if start_idx < total_rows:
            part_df = df.iloc[start_idx:end_idx]
            output_path = f"{base_path}_part{i+1:02d}{extension}"
            
            # 파일 형식에 따라 저장
            if extension == '.csv':
                part_df.to_csv(output_path, index=False)
            elif extension == '.xlsx':
                part_df.to_excel(output_path, index=False)
                
            print(f"Part {i+1:02d}: {len(part_df)}개 행 → {output_path}")

def main():
    if len(sys.argv) != 3:
        print("사용법: python split_csv.py <파일경로> <나눌개수>")
        print("예시: python split_csv.py data.csv 5")
        print("예시: python split_csv.py data.xlsx 3")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        num_parts = int(sys.argv[2])
        if num_parts <= 0:
            print("나눌 개수는 1 이상이어야 합니다.")
            sys.exit(1)
    except ValueError:
        print("나눌 개수는 정수여야 합니다.")
        sys.exit(1)
    
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        sys.exit(1)
    
    split_file(file_path, num_parts)
    print("완료!")

if __name__ == "__main__":
    main()