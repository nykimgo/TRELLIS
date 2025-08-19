#!/usr/bin/env python3
"""
FD_dinov2 Evaluator from Pre-rendered Images

This evaluator calculates Fréchet Distance using DINOv2 features from 
pre-rendered images instead of re-rendering 3D assets.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import argparse
import glob
from scipy.linalg import sqrtm
from torchvision import transforms

class FDDinoV2FromImages:
    """
    FD_dinov2 evaluator that works with pre-rendered images.
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the FD_dinov2 evaluator.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load DINOv2 model
        print(f"Loading DINOv2 model...")
        try:
            self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
            self.dinov2_model.eval().to(self.device)
            print("✓ DINOv2 model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load DINOv2 model: {e}")
            raise
        
        # Image preprocessing for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize((518, 518), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Initialized FD_dinov2 evaluator from images on {self.device}")
    
    def extract_middle_identifier(self, file_identifier: str) -> str:
        """
        Extract the middle part of file_identifier for directory naming.
        
        Args:
            file_identifier: String like "giraffe/giraffe_006/giraffe_006.blend"
            
        Returns:
            Middle part like "giraffe_006"
        """
        parts = file_identifier.split('/')
        if len(parts) >= 2:
            return parts[1]  # Return the middle part
        return file_identifier  # Fallback to original if splitting fails
    
    def load_sampled_data(self, sampled_csv_path: str) -> pd.DataFrame:
        """
        Load the 100 sampled assets data.
        
        Args:
            sampled_csv_path: Path to sampled_data_100_random.csv
            
        Returns:
            DataFrame with sampled asset data
        """
        if not os.path.exists(sampled_csv_path):
            raise FileNotFoundError(f"Sampled data file not found: {sampled_csv_path}")
        
        df = pd.read_csv(sampled_csv_path)
        print(f"Loaded {len(df)} sampled assets")
        return df
    
    def load_llm_results(self, results_excel_path: str) -> pd.DataFrame:
        """
        Load LLM-augmented prompt results from Excel file.
        
        Args:
            results_excel_path: Path to Excel file with LLM results
            
        Returns:
            DataFrame with LLM results
        """
        if not os.path.exists(results_excel_path):
            raise FileNotFoundError(f"Results file not found: {results_excel_path}")
        
        df = pd.read_excel(results_excel_path)
        print(f"Loaded {len(df)} LLM-augmented prompts")
        return df
    
    def find_rendered_images(self, base_path: str, middle_identifier: str, prefix: str) -> List[str]:
        """
        Find pre-rendered images for a specific asset.
        
        Args:
            base_path: Base path to rendered images
            middle_identifier: Middle part of file_identifier (e.g., "apple_011")
            prefix: Image prefix ("gt" or "gen")
            
        Returns:
            List of image file paths sorted by view angle
        """
        asset_dir = os.path.join(base_path, middle_identifier)
        if not os.path.exists(asset_dir):
            return []
        
        # Find images with pattern: {prefix}_view_{angle}.png
        pattern = os.path.join(asset_dir, f"{prefix}_view_*.png")
        image_files = glob.glob(pattern)

        # Sort by view angle
        def extract_angle(filepath):
            filename = os.path.basename(filepath)
            # Extract angle from filename like "gt_view_0.png" or "gen_view_45.png"
            try:
                angle_str = filename.split('_view_')[1].split('.')[0]
                return int(angle_str)
            except:
                return 0
        
        image_files.sort(key=extract_angle)
        return image_files
    
    def load_images_from_files(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Load images from file paths.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of loaded images as numpy arrays
        """
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        return images
    
    def extract_dinov2_features(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Extract DINOv2 features from rendered images.
        
        Args:
            images: List of rendered images as numpy arrays
            
        Returns:
            Tensor of DINOv2 features [num_images, feature_dim]
        """
        if not images:
            return torch.empty(0, 1024).to(self.device)  # DINOv2-L has 1024 features
        
        # Convert numpy arrays to PIL Images and preprocess
        processed_images = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            tensor_img = self.transform(pil_img)
            processed_images.append(tensor_img)
        
        # Stack and move to device
        batch_images = torch.stack(processed_images).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.dinov2_model(batch_images, is_training=True)
            # Use the CLS token features (global features)
            cls_features = features['x_norm_clstoken']  # [batch_size, 1024]
        
        return cls_features
    
    def calculate_frechet_distance(self, features_real: torch.Tensor, features_fake: torch.Tensor) -> float:
        """
        Calculate Fréchet Distance between two feature distributions.
        
        Args:
            features_real: Features from real/ground truth images [N, feature_dim]
            features_fake: Features from generated images [M, feature_dim]
            
        Returns:
            Fréchet Distance value
        """
        # Convert to numpy for scipy operations
        features_real = features_real.cpu().numpy()
        features_fake = features_fake.cpu().numpy()
        
        # Calculate means
        mu_real = np.mean(features_real, axis=0)
        mu_fake = np.mean(features_fake, axis=0)
        
        # Calculate covariances
        cov_real = np.cov(features_real, rowvar=False)
        cov_fake = np.cov(features_fake, rowvar=False)
        
        # Calculate FD
        diff = mu_real - mu_fake
        
        # Add small value to diagonal for numerical stability
        eps = 1e-6
        cov_real += eps * np.eye(cov_real.shape[0])
        cov_fake += eps * np.eye(cov_fake.shape[0])
        
        # Calculate sqrt of matrix product
        covmean = sqrtm(cov_real.dot(cov_fake))
        
        # Handle complex numbers from sqrtm
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m} too large in covmean")
            covmean = covmean.real
        
        # Calculate FD
        fd = np.sum((mu_real - mu_fake) ** 2) + np.trace(cov_real + cov_fake - 2 * covmean)
        
        return float(fd)
    
    def evaluate_single_asset_pair(self, 
                                  gt_base_path: str,
                                  gen_base_path: str,
                                  sampled_row: pd.Series, 
                                  llm_row: pd.Series) -> Dict:
        """
        Evaluate FD_dinov2 for a single asset pair using pre-rendered images.
        
        Args:
            gt_base_path: Base path to ground truth rendered images
            gen_base_path: Base path to generated asset rendered images
            sampled_row: Row from sampled data
            llm_row: Row from LLM results
            
        Returns:
            Dictionary with evaluation results
        """
        middle_identifier = self.extract_middle_identifier(sampled_row['file_identifier'])
        
        # Create generated asset path using LLM model name
        llm_model_dir = llm_row['llm_model'].replace(':', '_')
        gen_asset_path = os.path.join(gen_base_path, llm_model_dir)
        
        result = {
            'sha256': sampled_row['sha256'],
            'file_identifier': sampled_row['file_identifier'],
            'middle_identifier': middle_identifier,
            'object_name': llm_row['object_name_clean'],
            'llm_model': llm_row['category'],
            'gt_images_found': 0,
            'gen_images_found': 0,
            'real_features_extracted': False,
            'fake_features_extracted': False,
            'fd_dinov2': float('inf'),
            'success': False,
            'error': ''
        }
        
        try:
            # Find ground truth images
            gt_image_paths = self.find_rendered_images(gt_base_path, middle_identifier, "gt")
            if not gt_image_paths:
                result['error'] = f'No ground truth images found for {middle_identifier}'
                return result
            
            result['gt_images_found'] = len(gt_image_paths)
            
            # Find generated images
            gen_image_paths = self.find_rendered_images(gen_asset_path, middle_identifier, "gen")
            if not gen_image_paths:
                result['error'] = f'No generated images found for {middle_identifier} in {llm_model_dir}'
                return result
            
            result['gen_images_found'] = len(gen_image_paths)
            
            # Load images
            gt_images = self.load_images_from_files(gt_image_paths)
            gen_images = self.load_images_from_files(gen_image_paths)
            
            if not gt_images or not gen_images:
                result['error'] = 'Failed to load images'
                return result
            
            # Extract DINOv2 features
            gt_features = self.extract_dinov2_features(gt_images)
            gen_features = self.extract_dinov2_features(gen_images)
            
            if gt_features.shape[0] == 0 or gen_features.shape[0] == 0:
                result['error'] = 'Feature extraction failed'
                return result
            
            result['real_features_extracted'] = True
            result['fake_features_extracted'] = True
            
            # Calculate FD (need at least 2 samples for covariance)
            if gt_features.shape[0] >= 2 and gen_features.shape[0] >= 2:
                fd_value = self.calculate_frechet_distance(gt_features, gen_features)
                result['fd_dinov2'] = fd_value
                result['success'] = True
            else:
                result['error'] = f'Not enough samples for FD calculation (GT: {gt_features.shape[0]}, Gen: {gen_features.shape[0]}, need ≥2 each)'
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def evaluate_fd_dinov2(self, 
                          sampled_csv_path: str,
                          results_excel_path: str,
                          gt_base_path: str,
                          gen_base_path: str,
                          save_path: str = None,
                          max_assets: int = None,
                          llm_models_filter: List[str] = None) -> Dict:
        """
        Evaluate FD_dinov2 scores using pre-rendered images.
        
        Args:
            sampled_csv_path: Path to sampled_data_100_random.csv
            results_excel_path: Path to Excel file with LLM results
            gt_base_path: Base path to ground truth rendered images
            gen_base_path: Base path to generated asset rendered images
            save_path: Path to save results CSV file
            max_assets: Maximum number of assets to evaluate (for testing)
            llm_models_filter: List of LLM models to evaluate (None for all)
            
        Returns:
            Dictionary with aggregated results
        """
        # Load data
        sampled_df = self.load_sampled_data(sampled_csv_path)
        llm_df = self.load_llm_results(results_excel_path)
        
        # Filter by LLM models if specified
        if llm_models_filter:
            llm_df = llm_df[llm_df['category'].isin(llm_models_filter)]
            print(f"Filtered to {len(llm_df)} entries for models: {llm_models_filter}")
        
        # Match sampled data with LLM results by sha256
        print("Matching sampled data with LLM results...")
        matched_pairs = []
        
        for _, sampled_row in sampled_df.iterrows():
            sha256 = sampled_row['sha256']
            matching_llm_rows = llm_df[llm_df['sha256'] == sha256]
            
            for _, llm_row in matching_llm_rows.iterrows():
                matched_pairs.append((sampled_row, llm_row))
        
        print(f"Found {len(matched_pairs)} matched pairs")
        
        if max_assets:
            matched_pairs = matched_pairs[:max_assets]
            print(f"Limiting evaluation to first {max_assets} pairs")
        
        # Evaluate each pair
        evaluation_results = []
        successful_evaluations = 0
        total_fd_score = 0.0
        
        # Track results by LLM model
        model_results = {}
        
        for i, (sampled_row, llm_row) in enumerate(tqdm(matched_pairs, desc="Evaluating FD_dinov2 from images")):
            result = self.evaluate_single_asset_pair(gt_base_path, gen_base_path, sampled_row, llm_row)
            evaluation_results.append(result)
            
            llm_model = result['llm_model']
            if llm_model not in model_results:
                model_results[llm_model] = {'total': 0, 'successful': 0, 'total_fd': 0.0}
            
            model_results[llm_model]['total'] += 1
            
            if result['success']:
                successful_evaluations += 1
                fd_score = result['fd_dinov2']
                total_fd_score += fd_score
                model_results[llm_model]['successful'] += 1
                model_results[llm_model]['total_fd'] += fd_score
            
            # Print progress every 5 assets
            if (i + 1) % 5 == 0:
                current_success_rate = successful_evaluations / (i + 1)
                current_avg_fd = total_fd_score / successful_evaluations if successful_evaluations > 0 else float('inf')
                print(f"Progress: {i + 1}/{len(matched_pairs)}, "
                      f"Success rate: {current_success_rate:.2%}, "
                      f"Avg FD_dinov2: {current_avg_fd:.4f}")
        
        # Calculate aggregated metrics
        mean_fd_dinov2 = total_fd_score / successful_evaluations if successful_evaluations > 0 else float('inf')
        
        # Calculate per-model metrics
        model_summaries = {}
        for model, stats in model_results.items():
            if stats['successful'] > 0:
                model_avg_fd = stats['total_fd'] / stats['successful']
                model_summaries[model] = {
                    'total_assets': stats['total'],
                    'successful_evaluations': stats['successful'],
                    'success_rate': stats['successful'] / stats['total'],
                    'mean_fd_dinov2': model_avg_fd
                }
            else:
                model_summaries[model] = {
                    'total_assets': stats['total'],
                    'successful_evaluations': 0,
                    'success_rate': 0.0,
                    'mean_fd_dinov2': float('inf')
                }
        
        aggregated_results = {
            'mean_fd_dinov2': mean_fd_dinov2,
            'total_pairs': len(matched_pairs),
            'successful_evaluations': successful_evaluations,
            'success_rate': successful_evaluations / len(matched_pairs) if len(matched_pairs) > 0 else 0.0,
            'sampled_csv_path': sampled_csv_path,
            'results_excel_path': results_excel_path,
            'gt_base_path': gt_base_path,
            'gen_base_path': gen_base_path,
            'model_summaries': model_summaries
        }
        
        print(f"\n=== FD_dinov2 Evaluation Results (from pre-rendered images) ===")
        print(f"Sampled data: {sampled_csv_path}")
        print(f"LLM results: {results_excel_path}")
        print(f"GT images: {gt_base_path}")
        print(f"Generated images: {gen_base_path}")
        print(f"Total pairs: {len(matched_pairs)}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Success rate: {aggregated_results['success_rate']:.2%}")
        print(f"Mean FD_dinov2: {mean_fd_dinov2:.4f}")
        
        print(f"\n=== Per-Model Results ===")
        for model, summary in model_summaries.items():
            print(f"{model}: {summary['successful_evaluations']}/{summary['total_assets']} "
                  f"({summary['success_rate']:.1%}) - "
                  f"FD_dinov2: {summary['mean_fd_dinov2']:.4f}")
        
        # Save detailed results to CSV
        if save_path:
            df_results = pd.DataFrame(evaluation_results)
            df_results.to_csv(save_path, index=False)
            print(f"\nDetailed results saved to: {save_path}")
            
            # Also save summary
            summary_path = save_path.replace('.csv', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(aggregated_results, f, indent=2)
            print(f"Summary saved to: {summary_path}")
        
        return aggregated_results


def main():
    parser = argparse.ArgumentParser(
        description='FD_dinov2 Evaluation from Pre-rendered Images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 5 asset pairs
  python fd_dinov2_from_images.py --max_assets 5 --save_path test_fd_from_images.csv
  
  # Evaluate specific LLM models
  python fd_dinov2_from_images.py \\
    --llm_models "gemma3:27b-it-q8_0" "qwen3:14b-it-q4_0" \\
    --save_path model_fd_from_images.csv
  
  # Full evaluation
  python fd_dinov2_from_images.py --save_path full_fd_dinov2_from_images.csv
        """
    )
    
    parser.add_argument('--sampled_csv', 
                        type=str, 
                        default='/mnt/nas/tmp/nayeon/sampled_data_100_random.csv',
                        help='Path to sampled_data_100_random.csv file')
    
    parser.add_argument('--results_excel', 
                        type=str, 
                        default='/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx',
                        help='Path to Excel file with LLM results')
    
    parser.add_argument('--gt_base_path', 
                        type=str, 
                        default='/mnt/nas/Benchmark_Datatset/Toys4k/render_multiviews_for_CLIPeval',
                        help='Base path to ground truth rendered images')
    
    parser.add_argument('--gen_base_path', 
                        type=str, 
                        default='/mnt/nas/tmp/nayeon/output/CLIP_evaluation/TRELLIS-text-large',
                        help='Base path to generated asset rendered images')
    
    parser.add_argument('--save_path', 
                        type=str, 
                        default='fd_dinov2_from_images_results.csv',
                        help='Path to save detailed evaluation results CSV file')
    
    parser.add_argument('--max_assets', 
                        type=int, 
                        default=None,
                        help='Maximum number of asset pairs to evaluate (for testing)')
    
    parser.add_argument('--llm_models', 
                        type=str, 
                        nargs='*', 
                        default=None,
                        help='Specific LLM model categories to evaluate (e.g., "gemma3:27b-it-q8_0" "qwen3:14b-it-q4_0")')
    
    parser.add_argument('--device',
                        type=str,
                        default=None,
                        help='Device to use (cuda/cpu). Auto-detect if not specified.')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.sampled_csv):
        print(f"❌ Error: Sampled CSV file not found: {args.sampled_csv}")
        return
    
    if not os.path.exists(args.results_excel):
        print(f"❌ Error: Results Excel file not found: {args.results_excel}")
        return
    
    if not os.path.exists(args.gt_base_path):
        print(f"❌ Error: GT base path not found: {args.gt_base_path}")
        return
    
    if not os.path.exists(args.gen_base_path):
        print(f"❌ Error: Generated base path not found: {args.gen_base_path}")
        return
    
    print(f"📊 FD_dinov2 Evaluation from Pre-rendered Images")
    print(f"📁 Sampled CSV: {args.sampled_csv}")
    print(f"📁 LLM results: {args.results_excel}")
    print(f"📁 GT images: {args.gt_base_path}")
    print(f"📁 Generated images: {args.gen_base_path}")
    print(f"💾 Results will be saved to: {args.save_path}")
    if args.llm_models:
        print(f"🔍 Evaluating models: {', '.join(args.llm_models)}")
    if args.max_assets:
        print(f"⚡ Limited to {args.max_assets} asset pairs for testing")
    print()
    
    # Initialize evaluator
    evaluator = FDDinoV2FromImages(device=args.device)
    
    # Run evaluation
    results = evaluator.evaluate_fd_dinov2(
        args.sampled_csv, args.results_excel, args.gt_base_path, 
        args.gen_base_path, args.save_path, args.max_assets, args.llm_models
    )
    
    print(f"\n🎯 Final FD_dinov2: {results['mean_fd_dinov2']:.4f}")
    print(f"   (Lower values indicate better visual quality)")
    print(f"✅ Results saved to: {args.save_path}")


if __name__ == "__main__":
    main()