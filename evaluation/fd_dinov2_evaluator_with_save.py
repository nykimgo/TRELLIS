#!/usr/bin/env python3
"""
FD_dinov2 Evaluator with Saving Capabilities

This version saves rendered images and extracted features to disk.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import argparse
import ast
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh
from scipy.linalg import sqrtm
from torchvision import transforms
from fd_dinov2_evaluator import FDDinoV2Evaluator

class FDDinoV2EvaluatorWithSave(FDDinoV2Evaluator):
    """
    FD_dinov2 evaluator that saves rendered images and extracted features.
    """
    
    def __init__(self, device: str = None, save_renders: bool = False, save_features: bool = False):
        """
        Initialize the FD_dinov2 evaluator with saving capabilities.
        
        Args:
            device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
            save_renders: Whether to save rendered images
            save_features: Whether to save extracted features
        """
        super().__init__(device)
        self.save_renders = save_renders
        self.save_features = save_features
        
        if save_renders or save_features:
            print(f"✓ Saving enabled - Renders: {save_renders}, Features: {save_features}")
    
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
    
    def save_rendered_images(self, images: List[np.ndarray], save_dir: str, prefix: str):
        """
        Save rendered images to disk.
        
        Args:
            images: List of rendered images
            save_dir: Directory to save images
            prefix: Prefix for filenames (e.g., 'gt_giraffe' or 'gen_giraffe')
        """
        if not self.save_renders or not images:
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        for i, img in enumerate(images):
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            
            img_pil = Image.fromarray(img)
            filename = f"{prefix}_view_{self.yaw_angles[i]:03d}.png"
            img_path = os.path.join(save_dir, filename)
            img_pil.save(img_path)
        
        print(f"  💾 Saved {len(images)} rendered images to {save_dir}/{prefix}_*")
    
    def save_extracted_features(self, features: torch.Tensor, save_dir: str, prefix: str):
        """
        Save extracted DINOv2 features to disk.
        
        Args:
            features: Extracted features tensor
            save_dir: Directory to save features
            prefix: Prefix for filename (e.g., 'gt_giraffe' or 'gen_giraffe')
        """
        if not self.save_features or features.shape[0] == 0:
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as numpy array
        features_np = features.cpu().numpy()
        filename = f"{prefix}_dinov2_features.npy"
        features_path = os.path.join(save_dir, filename)
        np.save(features_path, features_np)
        
        # Also save as text for inspection
        filename_txt = f"{prefix}_dinov2_features.txt"
        features_txt_path = os.path.join(save_dir, filename_txt)
        with open(features_txt_path, 'w') as f:
            f.write(f"Shape: {features_np.shape}\n")
            f.write(f"Mean: {features_np.mean():.6f}\n")
            f.write(f"Std: {features_np.std():.6f}\n")
            f.write(f"Min: {features_np.min():.6f}\n")
            f.write(f"Max: {features_np.max():.6f}\n")
            f.write(f"Features (first 10 values per view):\n")
            for i in range(min(features_np.shape[0], 4)):
                f.write(f"View {i}: {features_np[i, :10].tolist()}\n")
        
        print(f"  💾 Saved features {features_np.shape} to {save_dir}/{prefix}_*")
    
    def evaluate_single_asset_pair(self, 
                                  dataset_path: str, 
                                  output_base_path: str,
                                  sampled_row: pd.Series, 
                                  llm_row: pd.Series,
                                  save_dir: str = None) -> Dict:
        """
        Evaluate FD_dinov2 for a single asset pair with optional saving.
        
        Args:
            dataset_path: Path to Toys4k dataset
            output_base_path: Path to generated assets
            sampled_row: Row from sampled data
            llm_row: Row from LLM results
            save_dir: Directory to save renders and features (if enabled)
            
        Returns:
            Dictionary with evaluation results
        """
        result = {
            'sha256': sampled_row['sha256'],
            'file_identifier': sampled_row['file_identifier'],
            'object_name': llm_row['object_name_clean'],
            'llm_model': llm_row['category'],
            'real_features_extracted': False,
            'fake_features_extracted': False,
            'fd_dinov2': float('inf'),
            'success': False,
            'error': '',
            'save_dir': save_dir if save_dir else ''
        }
        
        try:
            # Load ground truth asset
            gt_asset_path = self.find_asset_file(dataset_path, sampled_row['file_identifier'])
            if gt_asset_path is None:
                result['error'] = 'Ground truth asset file not found'
                return result
            
            gt_mesh = self.load_3d_asset(gt_asset_path)
            if gt_mesh is None:
                result['error'] = 'Failed to load ground truth mesh'
                return result
            
            # Load generated asset
            gen_asset_files = self.find_generated_assets(
                output_base_path, llm_row['category'], llm_row['object_name_clean']
            )
            if not gen_asset_files:
                result['error'] = 'Generated asset file not found'
                return result
            
            gen_mesh = self.load_3d_asset(gen_asset_files[0])
            if gen_mesh is None:
                result['error'] = 'Failed to load generated mesh'
                return result
            
            # Render both assets from 4 viewpoints
            gt_images = self.render_asset_4views(gt_mesh)
            gen_images = self.render_asset_4views(gen_mesh)
            
            if not gt_images or not gen_images:
                result['error'] = 'Rendering failed'
                return result
            
            # Save rendered images if enabled
            if save_dir and (self.save_renders or self.save_features):
                middle_identifier = self.extract_middle_identifier(sampled_row['file_identifier'])
                asset_save_dir = os.path.join(save_dir, middle_identifier)
                
                if self.save_renders:
                    self.save_rendered_images(gt_images, asset_save_dir, "gt")
                    self.save_rendered_images(gen_images, asset_save_dir, "gen")
            
            # Extract DINOv2 features
            gt_features = self.extract_dinov2_features(gt_images)
            gen_features = self.extract_dinov2_features(gen_images)
            
            if gt_features.shape[0] == 0 or gen_features.shape[0] == 0:
                result['error'] = 'Feature extraction failed'
                return result
            
            result['real_features_extracted'] = True
            result['fake_features_extracted'] = True
            
            # Save extracted features if enabled
            if save_dir and self.save_features:
                middle_identifier = self.extract_middle_identifier(sampled_row['file_identifier'])
                asset_save_dir = os.path.join(save_dir, middle_identifier)
                self.save_extracted_features(gt_features, asset_save_dir, "gt")
                self.save_extracted_features(gen_features, asset_save_dir, "gen")
            
            # Calculate FD (need at least 2 samples for covariance)
            if gt_features.shape[0] >= 2 and gen_features.shape[0] >= 2:
                fd_value = self.calculate_frechet_distance(gt_features, gen_features)
                result['fd_dinov2'] = fd_value
                result['success'] = True
            else:
                result['error'] = 'Not enough samples for FD calculation (need ≥2 views)'
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def evaluate_fd_dinov2(self, 
                          sampled_csv_path: str,
                          results_excel_path: str,
                          dataset_path: str,
                          output_base_path: str,
                          save_path: str = None,
                          max_assets: int = None,
                          llm_models_filter: List[str] = None,
                          save_dir: str = None) -> Dict:
        """
        Evaluate FD_dinov2 scores with optional saving of renders and features.
        
        Args:
            sampled_csv_path: Path to sampled_data_100_random.csv
            results_excel_path: Path to Excel file with LLM results
            dataset_path: Path to Toys4k dataset
            output_base_path: Base path to generated outputs directory
            save_path: Path to save results CSV file
            max_assets: Maximum number of assets to evaluate (for testing)
            llm_models_filter: List of LLM models to evaluate (None for all)
            save_dir: Directory to save renders and features (if enabled)
            
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
        
        # Create save directory if saving is enabled
        if save_dir and (self.save_renders or self.save_features):
            os.makedirs(save_dir, exist_ok=True)
            print(f"📁 Saving to: {save_dir}")
        
        # Evaluate each pair
        evaluation_results = []
        successful_evaluations = 0
        total_fd_score = 0.0
        
        # Track results by LLM model
        model_results = {}
        
        for i, (sampled_row, llm_row) in enumerate(tqdm(matched_pairs, desc="Evaluating FD_dinov2")):
            result = self.evaluate_single_asset_pair(dataset_path, output_base_path, sampled_row, llm_row, save_dir)
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
            'dataset_path': dataset_path,
            'output_base_path': output_base_path,
            'save_dir': save_dir if save_dir else '',
            'save_renders': self.save_renders,
            'save_features': self.save_features,
            'model_summaries': model_summaries
        }
        
        print(f"\n=== FD_dinov2 Evaluation Results ===")
        print(f"Sampled data: {sampled_csv_path}")
        print(f"LLM results: {results_excel_path}")
        print(f"Total pairs: {len(matched_pairs)}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Success rate: {aggregated_results['success_rate']:.2%}")
        print(f"Mean FD_dinov2: {mean_fd_dinov2:.4f}")
        
        print(f"\n=== Per-Model Results ===")
        for model, summary in model_summaries.items():
            print(f"{model}: {summary['successful_evaluations']}/{summary['total_assets']} "
                  f"({summary['success_rate']:.1%}) - "
                  f"FD_dinov2: {summary['mean_fd_dinov2']:.4f}")
        
        if save_dir and (self.save_renders or self.save_features):
            print(f"\n📁 Saved data to: {save_dir}")
        
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
        description='FD_dinov2 Evaluation with Saving Capabilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (no saving)
  python fd_dinov2_evaluator_with_save.py --max_assets 5
  
  # Save rendered images only
  python fd_dinov2_evaluator_with_save.py \\
    --max_assets 3 \\
    --save_renders \\
    --save_dir ./saved_renders
  
  # Save features only
  python fd_dinov2_evaluator_with_save.py \\
    --max_assets 3 \\
    --save_features \\
    --save_dir ./saved_features
  
  # Save both renders and features
  python fd_dinov2_evaluator_with_save.py \\
    --max_assets 3 \\
    --save_renders \\
    --save_features \\
    --save_dir ./saved_data
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
    
    parser.add_argument('--dataset_path', 
                        type=str, 
                        default='/mnt/nas/Benchmark_Datatset/Toys4k',
                        help='Path to Toys4k dataset directory')
    
    parser.add_argument('--output_base_path', 
                        type=str, 
                        default='/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816',
                        help='Base path to directory containing generated 3D assets')
    
    parser.add_argument('--save_path', 
                        type=str, 
                        default='fd_dinov2_with_save_results.csv',
                        help='Path to save detailed evaluation results CSV file')
    
    parser.add_argument('--max_assets', 
                        type=int, 
                        default=None,
                        help='Maximum number of asset pairs to evaluate (for testing)')
    
    parser.add_argument('--llm_models', 
                        type=str, 
                        nargs='*', 
                        default=None,
                        help='Specific LLM model categories to evaluate (e.g., gemma3 qwen3)')
    
    parser.add_argument('--device',
                        type=str,
                        default=None,
                        help='Device to use (cuda/cpu). Auto-detect if not specified.')
    
    parser.add_argument('--save_renders',
                        action='store_true',
                        help='Save rendered 4-view images to disk')
    
    parser.add_argument('--save_features',
                        action='store_true',
                        help='Save extracted DINOv2 features to disk')
    
    parser.add_argument('--save_dir',
                        type=str,
                        default='./fd_saved_data',
                        help='Directory to save renders and features')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.sampled_csv):
        print(f"❌ Error: Sampled CSV file not found: {args.sampled_csv}")
        return
    
    if not os.path.exists(args.results_excel):
        print(f"❌ Error: Results Excel file not found: {args.results_excel}")
        return
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ Error: Dataset path not found: {args.dataset_path}")
        return
    
    if not os.path.exists(args.output_base_path):
        print(f"❌ Error: Output base path not found: {args.output_base_path}")
        return
    
    print(f"📊 FD_dinov2 Evaluation with Saving Capabilities")
    print(f"📁 Sampled CSV: {args.sampled_csv}")
    print(f"📁 LLM results: {args.results_excel}")
    print(f"📁 Dataset path: {args.dataset_path}")
    print(f"📁 Generated assets: {args.output_base_path}")
    print(f"💾 Results will be saved to: {args.save_path}")
    if args.save_renders or args.save_features:
        print(f"📁 Data will be saved to: {args.save_dir}")
        print(f"  - Save renders: {args.save_renders}")
        print(f"  - Save features: {args.save_features}")
    if args.llm_models:
        print(f"🔍 Evaluating models: {', '.join(args.llm_models)}")
    if args.max_assets:
        print(f"⚡ Limited to {args.max_assets} asset pairs for testing")
    print()
    
    # Initialize evaluator with saving options
    evaluator = FDDinoV2EvaluatorWithSave(
        device=args.device, 
        save_renders=args.save_renders, 
        save_features=args.save_features
    )
    
    # Run evaluation
    results = evaluator.evaluate_fd_dinov2(
        args.sampled_csv, args.results_excel, args.dataset_path, 
        args.output_base_path, args.save_path, args.max_assets, 
        args.llm_models, args.save_dir if (args.save_renders or args.save_features) else None
    )
    
    print(f"\n🎯 Final FD_dinov2: {results['mean_fd_dinov2']:.4f}")
    print(f"   (Lower values indicate better visual quality)")
    print(f"✅ Results saved to: {args.save_path}")


if __name__ == "__main__":
    main()