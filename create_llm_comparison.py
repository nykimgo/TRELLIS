#!/usr/bin/env python3
"""
LLM 모델별 3D 생성 결과 비교 이미지 생성 스크립트
"""

import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import re
from typing import Dict, List, Optional, Tuple
import logging

class LLMComparisonGenerator:
    def __init__(self, base_output_dir: str = "/mnt/nas/tmp/nayeon"):
        self.base_output_dir = Path(base_output_dir)
        self.thumbnail_suffixes = ["004s.jpg", "005s.jpg", "006s.jpg"]
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def load_model_filter(self, filter_file: str) -> List[str]:
        """
        txt 파일에서 비교할 LLM 모델 목록 로드
        """
        try:
            with open(filter_file, 'r', encoding='utf-8') as f:
                models = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(models)} models from filter file: {models}")
            return models
        except Exception as e:
            self.logger.error(f"Failed to load model filter file {filter_file}: {e}")
            return []
    
    def find_object_results(self, trellis_model: str, date: str, object_name: str, filter_models: Optional[List[str]] = None) -> Dict[str, Dict[str, List[str]]]:
        """
        특정 객체에 대한 모든 LLM 모델별 썸네일 결과 찾기
        
        Returns:
            {llm_model: {suffix: [file_paths]}}
        """
        results = {}
        
        # 기본 출력 디렉토리 구조
        base_path = self.base_output_dir / "outputs" / trellis_model / date
        
        if not base_path.exists():
            self.logger.error(f"Base path does not exist: {base_path}")
            return results
        
        # LLM 모델 디렉토리들 탐색
        for llm_dir in base_path.iterdir():
            if not llm_dir.is_dir():
                continue
                
            llm_model = llm_dir.name
            
            # 필터링 체크 (filter_models가 지정된 경우)
            if filter_models is not None and llm_model not in filter_models:
                continue
                
            object_dir = llm_dir / object_name
            
            if not object_dir.exists() or not object_dir.is_dir():
                continue
            
            # 썸네일 파일들 찾기
            llm_results = {}
            for suffix in self.thumbnail_suffixes:
                # 패턴: {object_name}_{trellis_model}_{llm_model}_{seed}_gs_{suffix}
                pattern = f"{object_name}_{trellis_model}_{llm_model}_*_gs_{suffix}"
                matching_files = list(object_dir.glob(pattern))
                
                if matching_files:
                    llm_results[suffix] = [str(f) for f in matching_files]
                else:
                    llm_results[suffix] = []
            
            if any(llm_results.values()):  # 하나라도 파일이 있으면 추가
                results[llm_model] = llm_results
        
        return results
    
    def create_comparison_image(self, 
                              trellis_model: str, 
                              object_name: str, 
                              results: Dict[str, Dict[str, List[str]]],
                              output_path: str) -> bool:
        """
        비교 이미지 생성
        """
        if not results:
            self.logger.error("No results found to create comparison image")
            return False
        
        # 이미지 설정
        thumbnail_size = (256, 256)  # 썸네일 크기
        margin = 20
        title_height = 60  # 줄임
        subtitle_height = 30  # 줄임
        llm_label_width = 200
        
        # LLM 모델 수와 열 수 계산
        num_llm_models = len(results)
        num_cols = len(self.thumbnail_suffixes)
        
        # 전체 이미지 크기 계산
        img_width = llm_label_width + (thumbnail_size[0] + margin) * num_cols + margin
        img_height = title_height + subtitle_height + (thumbnail_size[1] + margin) * num_llm_models + margin * 2
        
        # 빈 이미지 생성 (흰색 배경)
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 폰트 설정 (기본 폰트 사용)
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            label_font = ImageFont.load_default()
        
        # 제목 그리기
        title_text = f"LLM Prompt-based {object_name.replace('_', ' ').title()} Output Comparison"
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (img_width - title_width) // 2
        draw.text((title_x, margin), title_text, fill='black', font=title_font)
        
        # 부제목 그리기
        subtitle_text = f"TRELLIS Model: {trellis_model}"
        subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (img_width - subtitle_width) // 2
        draw.text((subtitle_x, margin + title_height), subtitle_text, fill='gray', font=subtitle_font)
        
        # LLM 모델별 썸네일 배치 (헤더 제거)
        start_y = title_height + subtitle_height + margin
        
        for row_idx, (llm_model, llm_results) in enumerate(results.items()):
            y_pos = start_y + row_idx * (thumbnail_size[1] + margin)
            
            # LLM 모델명 레이블 (두 줄로 표시, 크기 조정)
            label_text = llm_model.replace('_', ':')  # gpt-oss_20b -> gpt-oss:20b
            
            if ':' in label_text:
                # ':' 기준으로 분할하여 두 줄로 표시
                parts = label_text.split(':', 1)
                first_line = parts[0]
                second_line = f":{parts[1]}"
                
                # 첫 번째 줄 (title 크기)
                draw.text((margin, y_pos + thumbnail_size[1] // 2 - 15), first_line, fill='black', font=title_font)
                # 두 번째 줄 (subtitle 크기)
                draw.text((margin, y_pos + thumbnail_size[1] // 2 + 10), second_line, fill='black', font=subtitle_font)
            else:
                # ':' 없는 경우 한 줄로 표시 (title 크기)
                draw.text((margin, y_pos + thumbnail_size[1] // 2), label_text, fill='black', font=title_font)
            
            # 썸네일 이미지들 배치
            for col_idx, suffix in enumerate(self.thumbnail_suffixes):
                x_pos = llm_label_width + col_idx * (thumbnail_size[0] + margin)
                
                if suffix in llm_results and llm_results[suffix]:
                    # 첫 번째 썸네일 사용 (여러 개가 있을 경우)
                    thumbnail_path = llm_results[suffix][0]
                    
                    try:
                        # 썸네일 로드 및 리사이즈
                        thumbnail = Image.open(thumbnail_path)
                        thumbnail = thumbnail.resize(thumbnail_size, Image.Resampling.LANCZOS)
                        
                        # 이미지 붙여넣기
                        img.paste(thumbnail, (x_pos, y_pos))
                        
                        # 테두리 그리기
                        draw.rectangle([x_pos, y_pos, x_pos + thumbnail_size[0], y_pos + thumbnail_size[1]], 
                                     outline='lightgray', width=1)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load thumbnail {thumbnail_path}: {e}")
                        # 빈 박스 그리기
                        draw.rectangle([x_pos, y_pos, x_pos + thumbnail_size[0], y_pos + thumbnail_size[1]], 
                                     outline='red', fill='lightgray', width=2)
                        draw.text((x_pos + 10, y_pos + thumbnail_size[1] // 2), "Missing", fill='red', font=label_font)
                else:
                    # 결과가 없는 경우 빈 박스
                    draw.rectangle([x_pos, y_pos, x_pos + thumbnail_size[0], y_pos + thumbnail_size[1]], 
                                 outline='lightgray', fill='whitesmoke', width=1)
                    draw.text((x_pos + 10, y_pos + thumbnail_size[1] // 2), "No Result", fill='gray', font=label_font)
        
        # 이미지 저장
        try:
            img.save(output_path, 'PNG', quality=95)
            self.logger.info(f"Comparison image saved: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save image: {e}")
            return False
    
    def generate_comparison(self, 
                          trellis_model: str, 
                          date: str, 
                          object_name: str, 
                          output_dir: Optional[str] = None,
                          filter_file: Optional[str] = None) -> bool:
        """
        특정 객체에 대한 LLM 모델별 비교 이미지 생성
        """
        # 모델 필터 로드
        filter_models = None
        if filter_file:
            filter_models = self.load_model_filter(filter_file)
            if not filter_models:
                self.logger.error("No valid models found in filter file")
                return False
        
        # 결과 찾기
        self.logger.info(f"Searching for {object_name} results in {trellis_model}/{date}")
        if filter_models:
            self.logger.info(f"Filtering for models: {filter_models}")
        results = self.find_object_results(trellis_model, date, object_name, filter_models)
        
        if not results:
            self.logger.error(f"No results found for {object_name}")
            return False
        
        self.logger.info(f"Found results for {len(results)} LLM models: {list(results.keys())}")
        
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = self.base_output_dir / "outputs" / trellis_model / date
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 출력 파일명
        output_filename = f"Comparison_LLMmodel_{object_name}.png"
        output_path = output_dir / output_filename
        
        # 비교 이미지 생성
        success = self.create_comparison_image(trellis_model, object_name, results, str(output_path))
        
        if success:
            self.logger.info(f"✅ Comparison image created successfully: {output_path}")
        else:
            self.logger.error("❌ Failed to create comparison image")
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Create LLM model comparison images for 3D generation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create comparison for BlockyRobot with TRELLIS-text-large model
  python create_llm_comparison.py --trellis_model TRELLIS-text-large --date 20250814 --object BlockyRobot
  
  # Compare only specific LLM models using filter file
  python create_llm_comparison.py --trellis_model TRELLIS-text-large --date 20250814 --object BlockyRobot --filter_file selected_models.txt
  
  # Specify custom output directory
  python create_llm_comparison.py --trellis_model TRELLIS-text-large --date 20250814 --object BlockyRobot --output_dir /custom/path
  
  # Use different base directory
  python create_llm_comparison.py --trellis_model TRELLIS-text-large --date 20250814 --object BlockyRobot --base_dir /different/base
        """
    )
    
    parser.add_argument(
        '--trellis_model',
        type=str,
        required=True,
        help='TRELLIS model name (e.g., TRELLIS-text-large)'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        required=True,
        help='Date directory (e.g., 20250814)'
    )
    
    parser.add_argument(
        '--object',
        type=str,
        required=True,
        help='Object name to compare (e.g., BlockyRobot)'
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/mnt/nas/tmp/nayeon',
        help='Base output directory (default: /mnt/nas/tmp/nayeon)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Specific output directory for the comparison image (optional)'
    )
    
    parser.add_argument(
        '--filter_file',
        type=str,
        help='Text file containing LLM model names to compare (one per line)'
    )
    
    args = parser.parse_args()
    
    # 비교 이미지 생성기 초기화
    generator = LLMComparisonGenerator(base_output_dir=args.base_dir)
    
    # 비교 이미지 생성
    success = generator.generate_comparison(
        trellis_model=args.trellis_model,
        date=args.date,
        object_name=args.object,
        output_dir=args.output_dir,
        filter_file=args.filter_file
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())