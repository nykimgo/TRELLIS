#!/usr/bin/env python3
"""
배치로 모든 객체에 대한 LLM 모델 비교 이미지 생성
"""

import os
import subprocess
import argparse
from pathlib import Path
import logging


def get_unique_objects(base_dir: str, trellis_model: str, date: str) -> set:
    """
    지정된 디렉토리에서 모든 고유 객체명 추출
    """
    objects = set()
    base_path = Path(base_dir) / "outputs" / trellis_model / date
    
    if not base_path.exists():
        return objects
    
    # 모든 LLM 모델 디렉토리 탐색
    for llm_dir in base_path.iterdir():
        if llm_dir.is_dir() and not llm_dir.name.startswith('Comparison_'):
            # 각 LLM 디렉토리 안의 객체들 수집
            for obj_dir in llm_dir.iterdir():
                if obj_dir.is_dir():
                    objects.add(obj_dir.name)
    
    return objects


def main():
    parser = argparse.ArgumentParser(description="Batch create LLM comparison images for all objects")
    parser.add_argument('--trellis_model', required=True, help='TRELLIS model name')
    parser.add_argument('--date', required=True, help='Date directory')
    parser.add_argument('--base_dir', default='/mnt/nas/tmp/nayeon', help='Base directory')
    parser.add_argument('--filter_file', help='Text file containing LLM model names to compare')
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 모든 고유 객체 찾기
    objects = get_unique_objects(args.base_dir, args.trellis_model, args.date)
    
    if not objects:
        logger.error(f"No objects found in {args.base_dir}/outputs/{args.trellis_model}/{args.date}")
        return 1
    
    logger.info(f"Found {len(objects)} unique objects: {sorted(objects)}")
    
    # 각 객체에 대해 비교 이미지 생성
    success_count = 0
    total_count = len(objects)
    
    for obj in sorted(objects):
        logger.info(f"Creating comparison for {obj}...")
        
        try:
            # create_llm_comparison.py 실행
            cmd = [
                "python", "create_llm_comparison.py",
                "--trellis_model", args.trellis_model,
                "--date", args.date,
                "--object", obj,
                "--base_dir", args.base_dir
            ]
            
            # 필터 파일이 지정된 경우 추가
            if args.filter_file:
                cmd.extend(["--filter_file", args.filter_file])
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            success_count += 1
            logger.info(f"✅ Successfully created comparison for {obj}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to create comparison for {obj}: {e}")
            if e.stdout:
                logger.error(f"STDOUT: {e.stdout}")
            if e.stderr:
                logger.error(f"STDERR: {e.stderr}")
    
    logger.info(f"Batch processing completed: {success_count}/{total_count} successful")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    exit(main())