#!/usr/bin/env python3
"""
FD_dinov2 Evaluator with Blender Rendering
Evaluates Fréchet Distance between DINOv2 features of rendered images.
Uses TRELLIS's Blender rendering system for high-quality texture rendering.
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from scipy.linalg import sqrtm
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class FD_dinov2_BlenderEvaluator:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.setup_dinov2_model()
        self.setup_blender()
        
    def setup_dinov2_model(self):
        """Initialize DINOv2 model"""
        print("Loading DINOv2 model...")
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        self.dinov2_model.eval()
        self.dinov2_model.to(self.device)
        
        # Transform for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def setup_blender(self):
        """Setup Blender paths and installation"""
        self.blender_path = '/tmp/blender-3.0.1-linux-x64/blender'
        self.blender_script = '/home/sr/TRELLIS/dataset_toolkits/blender_script/render.py'
        
        # Check if Blender is installed
        if not os.path.exists(self.blender_path):
            print("Installing Blender...")
            self._install_blender()
            
    def _install_blender(self):
        """Install Blender if not present"""
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system('wget https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz -P /tmp')
        os.system('tar -xvf /tmp/blender-3.0.1-linux-x64.tar.xz -C /tmp')
        
    def render_4_views_blender(self, model_path: str, output_dir: str) -> List[str]:
        """
        Render 4 views using Blender with texture support
        Returns list of rendered image paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Define 4 views: 0°, 90°, 180°, 270° yaw angles with 30° pitch
        views = [
            {'yaw': 0 * np.pi / 180, 'pitch': 30 * np.pi / 180, 'radius': 2, 'fov': 40 * np.pi / 180},
            {'yaw': 90 * np.pi / 180, 'pitch': 30 * np.pi / 180, 'radius': 2, 'fov': 40 * np.pi / 180},
            {'yaw': 180 * np.pi / 180, 'pitch': 30 * np.pi / 180, 'radius': 2, 'fov': 40 * np.pi / 180},
            {'yaw': 270 * np.pi / 180, 'pitch': 30 * np.pi / 180, 'radius': 2, 'fov': 40 * np.pi / 180}
        ]
        
        # Prepare Blender command
        args = [
            self.blender_path, '-b', '-P', self.blender_script,
            '--',
            '--views', json.dumps(views),
            '--object', os.path.expanduser(model_path),
            '--resolution', '512',
            '--output_folder', output_dir,
            '--engine', 'CYCLES'
        ]
        
        # Handle .blend files
        if model_path.endswith('.blend'):
            args.insert(1, model_path)
            
        # Run Blender rendering
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"Blender rendering failed: {result.stderr}")
                return []
        except subprocess.TimeoutExpired:
            print(f"Blender rendering timed out for {model_path}")
            return []
            
        # Collect rendered images
        rendered_images = []
        for i in range(4):
            img_path = os.path.join(output_dir, f'{i:03d}.png')
            if os.path.exists(img_path):
                rendered_images.append(img_path)
                
        return rendered_images
        
    def extract_dinov2_features(self, image_paths: List[str]) -> torch.Tensor:
        """Extract DINOv2 features from rendered images"""
        processed_images = []
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"Warning: Image not found: {img_path}")
                continue
                
            # Load and process image
            pil_img = Image.open(img_path).convert('RGB')
            tensor_img = self.transform(pil_img)
            processed_images.append(tensor_img)
        
        if not processed_images:
            return torch.empty(0, 1024).to(self.device)
            
        batch_images = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            features = self.dinov2_model(batch_images, is_training=True)
            cls_features = features['x_norm_clstoken']
        
        return cls_features
        
    def calculate_frechet_distance(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Calculate Fréchet Distance between two sets of features"""
        # Convert to numpy and calculate statistics
        mu1 = features1.mean(dim=0).cpu().numpy()
        mu2 = features2.mean(dim=0).cpu().numpy()
        
        sigma1 = torch.cov(features1.T).cpu().numpy()
        sigma2 = torch.cov(features2.T).cpu().numpy()
        
        # Calculate Fréchet Distance
        diff = mu1 - mu2
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-6
        sigma1 += eps * np.eye(sigma1.shape[0])
        sigma2 += eps * np.eye(sigma2.shape[0])
        
        # Compute square root of product of covariance matrices
        covmean = sqrtm(sigma1.dot(sigma2))
        
        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fd = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
        
        return float(fd)
        
    def get_or_create_gt_cache(self, gt_path: str, gt_cache_dir: str, base_asset_name: str, sha256: str) -> Tuple[List[str], torch.Tensor]:
        """
        Get cached GT views and features, or create them if they don't exist
        Returns: (gt_image_paths, gt_features)
        """
        # Create cache directory structure
        cache_asset_dir = os.path.join(gt_cache_dir, f"{base_asset_name}_{sha256[:6]}")
        
        # Check if cache exists
        cache_features_path = os.path.join(cache_asset_dir, 'gt_features.npy')
        cache_images = []
        for i in range(4):
            img_path = os.path.join(cache_asset_dir, f'gt_view_{i}.png')
            cache_images.append(img_path)
            
        # If all cached files exist, load them
        if os.path.exists(cache_features_path) and all(os.path.exists(img) for img in cache_images):
            print(f"Loading cached GT views for {base_asset_name}")
            gt_features = torch.from_numpy(np.load(cache_features_path)).to(self.device)
            return cache_images, gt_features
        
        # Otherwise, render and cache
        print(f"Rendering and caching GT views for {base_asset_name}")
        os.makedirs(cache_asset_dir, exist_ok=True)
        
        # Render GT views
        with tempfile.TemporaryDirectory() as temp_dir:
            gt_render_dir = os.path.join(temp_dir, 'gt_renders')
            gt_images = self.render_4_views_blender(gt_path, gt_render_dir)
            
            if len(gt_images) == 0:
                return [], torch.empty(0, 1024).to(self.device)
            
            # Extract features
            gt_features = self.extract_dinov2_features(gt_images)
            
            # Save to cache
            for i, gt_img in enumerate(gt_images):
                cache_img_path = os.path.join(cache_asset_dir, f'gt_view_{i}.png')
                os.system(f'cp {gt_img} {cache_img_path}')
                
            np.save(cache_features_path, gt_features.cpu().numpy())
            
            return cache_images, gt_features
    
    def evaluate_single_asset(self, gen_path: str, gt_path: str, save_dir: Optional[str] = None, asset_name: str = None, 
                             gt_cache_dir: str = None, base_asset_name: str = None, sha256: str = None) -> Dict:
        """Evaluate a single generated asset against ground truth"""
        if asset_name is None:
            asset_name = Path(gen_path).stem
        
        # Create temporary directory for generated asset rendering
        with tempfile.TemporaryDirectory() as temp_dir:
            gen_render_dir = os.path.join(temp_dir, 'gen_renders')
            
            # Render generated asset
            print(f"Rendering generated asset: {asset_name}")
            gen_images = self.render_4_views_blender(gen_path, gen_render_dir)
            
            # Get or create cached GT views and features
            if gt_cache_dir and base_asset_name and sha256:
                gt_images, gt_features = self.get_or_create_gt_cache(gt_path, gt_cache_dir, base_asset_name, sha256)
            else:
                # Fallback to direct rendering if cache not available
                print(f"Rendering ground truth asset: {asset_name}")
                gt_render_dir = os.path.join(temp_dir, 'gt_renders')
                gt_images = self.render_4_views_blender(gt_path, gt_render_dir)
                gt_features = self.extract_dinov2_features(gt_images) if gt_images else torch.empty(0, 1024).to(self.device)
            
            if len(gen_images) == 0 or len(gt_images) == 0:
                print(f"Warning: Failed to render {asset_name}")
                return {
                    'asset_name': asset_name,
                    'fd_dinov2': float('inf'),
                    'gen_rendered': len(gen_images),
                    'gt_rendered': len(gt_images),
                    'error': 'Rendering failed'
                }
            
            # Save rendered images if requested
            if save_dir:
                save_asset_dir = save_dir
                os.makedirs(save_asset_dir, exist_ok=True)
                
                # Copy rendered images
                for i, (gen_img, gt_img) in enumerate(zip(gen_images, gt_images)):
                    gen_save_path = os.path.join(save_asset_dir, f'gen_view_{i}.png')
                    gt_save_path = os.path.join(save_asset_dir, f'gt_view_{i}.png')
                    os.system(f'cp {gen_img} {gen_save_path}')
                    os.system(f'cp {gt_img} {gt_save_path}')
            
            # Extract features for generated images only (GT features already extracted if cached)
            print(f"Extracting DINOv2 features for {asset_name}")
            gen_features = self.extract_dinov2_features(gen_images)
            
            # GT features already extracted in cache or fallback above
            
            if gen_features.size(0) == 0 or gt_features.size(0) == 0:
                return {
                    'asset_name': asset_name,
                    'fd_dinov2': float('inf'),
                    'gen_rendered': len(gen_images),
                    'gt_rendered': len(gt_images),
                    'error': 'Feature extraction failed'
                }
            
            # Calculate Fréchet Distance
            fd_score = self.calculate_frechet_distance(gen_features, gt_features)
            
            # Save features if requested
            if save_dir:
                np.save(os.path.join(save_dir, 'gen_features.npy'), gen_features.cpu().numpy())
                np.save(os.path.join(save_dir, 'gt_features.npy'), gt_features.cpu().numpy())
                
                # Save feature statistics
                with open(os.path.join(save_dir, 'feature_stats.txt'), 'w') as f:
                    f.write(f"Generated features shape: {gen_features.shape}\n")
                    f.write(f"Ground truth features shape: {gt_features.shape}\n")
                    f.write(f"FD_dinov2 score: {fd_score}\n")
            
            return {
                'asset_name': asset_name,
                'fd_dinov2': fd_score,
                'gen_rendered': len(gen_images),
                'gt_rendered': len(gt_images)
            }
    
    def evaluate_dataset(self, generated_dir: str, ground_truth_dir: str, 
                        sampled_csv: str, results_excel: str, output_path: str, 
                        save_renders: bool = False, max_assets: int = None) -> pd.DataFrame:
        """
        Evaluate FD_dinov2 scores for a dataset
        
        Args:
            generated_dir: Directory containing generated 3D assets
            ground_truth_dir: Directory containing ground truth 3D assets  
            sampled_csv: CSV file with sampled asset names
            results_excel: Excel file with LLM generation results
            output_path: Path to save evaluation results
            save_renders: Whether to save rendered images and features
            max_assets: Maximum number of assets to evaluate (for testing)
        """
        
        # Load sampled data
        sampled_df = pd.read_csv(sampled_csv)
        
        # Load LLM results Excel file
        results_df = pd.read_excel(results_excel)
        
        # Match sampled CSV with LLM results by sha256
        merged_df = sampled_df.merge(results_df, on='sha256', how='inner')
        print(f"Matched {len(merged_df)} assets between sampled CSV and LLM results")
        
        # Limit number of assets if specified (but keep all LLM model variations)
        if max_assets is not None and max_assets > 0:
            merged_df = merged_df.head(max_assets)
            print(f"Limited to {max_assets} LLM results for testing")
        
        results = []
        save_dir = None
        if save_renders:
            save_dir = os.path.splitext(output_path)[0] + '_renders'
            os.makedirs(save_dir, exist_ok=True)
            
        # Setup GT cache directory
        gt_cache_dir = os.path.join(ground_truth_dir, 'render_4views_for_fd')
        os.makedirs(gt_cache_dir, exist_ok=True)
        
        print(f"Evaluating {len(merged_df)} assets...")
        
        for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
            file_identifier = row['file_identifier_x']
            # Extract base asset name from file_identifier (e.g., giraffe/giraffe_006/giraffe_006.blend -> giraffe_006)
            base_asset_name = Path(file_identifier).stem
            
            # Get LLM model and object name from results
            llm_model = row['llm_model']
            object_name_clean = row['object_name_clean']
            sha256 = row['sha256']
            
            # Convert LLM model name (: -> _)
            llm_model_folder = llm_model.replace(':', '_')
            
            # Create unique asset name including LLM model info
            asset_name = f"{base_asset_name}_{llm_model_folder}_{sha256[:6]}"
            
            # Find generated asset in proper directory structure
            gen_asset_dir = os.path.join(generated_dir, llm_model_folder, object_name_clean)
            
            gen_path = None
            if os.path.exists(gen_asset_dir):
                # Look for any .glb or .ply file in the directory
                for file in os.listdir(gen_asset_dir):
                    if file.endswith(('.glb', '.ply')):
                        gen_path = os.path.join(gen_asset_dir, file)
                        break
                    
            if gen_path is None:
                print(f"Generated asset not found for {asset_name}")
                print(f"  Looking in: {gen_asset_dir}")
                if os.path.exists(gen_asset_dir):
                    print(f"  Directory exists, contents: {os.listdir(gen_asset_dir)}")
                else:
                    print(f"  Directory does not exist")
                results.append({
                    'asset_name': asset_name,
                    'base_asset_name': base_asset_name,
                    'llm_model': llm_model,
                    'object_name_clean': object_name_clean,
                    'sha256': sha256,
                    'fd_dinov2': float('inf'),
                    'gen_rendered': 0,
                    'gt_rendered': 0,
                    'error': 'Generated asset not found'
                })
                continue
            else:
                print(f"Found generated asset: {gen_path}")
                
            # Find ground truth asset using file_identifier
            gt_path = os.path.join(ground_truth_dir, file_identifier)
            if not os.path.exists(gt_path):
                gt_path = None
                    
            if gt_path is None:
                print(f"Ground truth asset not found for {asset_name}")
                print(f"  Looking for: {os.path.join(ground_truth_dir, file_identifier)}")
                results.append({
                    'asset_name': asset_name,
                    'base_asset_name': base_asset_name,
                    'llm_model': llm_model,
                    'object_name_clean': object_name_clean,
                    'sha256': sha256,
                    'fd_dinov2': float('inf'),
                    'gen_rendered': 0,
                    'gt_rendered': 0,
                    'error': 'Ground truth asset not found'
                })
                continue
            
            # Evaluate this asset
            try:
                # Create folder name with model info and sha256
                folder_name = f"{object_name_clean}_{sha256[:6]}"
                model_save_dir = None
                if save_dir:
                    model_save_dir = os.path.join(save_dir, "TRELLIS-text-large", llm_model_folder, folder_name)
                
                result = self.evaluate_single_asset(gen_path, gt_path, model_save_dir, asset_name, 
                                                   gt_cache_dir, base_asset_name, sha256)
                # Add LLM model info to result
                result.update({
                    'base_asset_name': base_asset_name,
                    'llm_model': llm_model,
                    'object_name_clean': object_name_clean,
                    'sha256': sha256
                })
                results.append(result)
            except Exception as e:
                print(f"Error evaluating {asset_name}: {str(e)}")
                results.append({
                    'asset_name': asset_name,
                    'base_asset_name': base_asset_name,
                    'llm_model': llm_model,
                    'object_name_clean': object_name_clean,
                    'sha256': sha256,
                    'fd_dinov2': float('inf'),
                    'gen_rendered': 0,
                    'gt_rendered': 0,
                    'error': str(e)
                })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        valid_scores = results_df[results_df['fd_dinov2'] != float('inf')]['fd_dinov2']
        
        summary = {
            'overall': {
                'total_assets': len(results_df),
                'successful_evaluations': len(valid_scores),
                'failed_evaluations': len(results_df) - len(valid_scores),
                'mean_fd_dinov2': valid_scores.mean() if len(valid_scores) > 0 else float('inf'),
                'std_fd_dinov2': valid_scores.std() if len(valid_scores) > 0 else float('inf'),
                'median_fd_dinov2': valid_scores.median() if len(valid_scores) > 0 else float('inf'),
                'min_fd_dinov2': valid_scores.min() if len(valid_scores) > 0 else float('inf'),
                'max_fd_dinov2': valid_scores.max() if len(valid_scores) > 0 else float('inf')
            }
        }
        
        # Calculate LLM model-specific statistics
        llm_model_stats = {}
        valid_results_df = results_df[results_df['fd_dinov2'] != float('inf')]
        
        if len(valid_results_df) > 0:
            for llm_model in valid_results_df['llm_model'].unique():
                model_data = valid_results_df[valid_results_df['llm_model'] == llm_model]['fd_dinov2']
                if len(model_data) > 0:
                    llm_model_stats[llm_model] = {
                        'count': len(model_data),
                        'mean_fd_dinov2': model_data.mean(),
                        'std_fd_dinov2': model_data.std(),
                        'median_fd_dinov2': model_data.median(),
                        'min_fd_dinov2': model_data.min(),
                        'max_fd_dinov2': model_data.max()
                    }
        
        summary['by_llm_model'] = llm_model_stats
        
        print("\n=== FD_dinov2 Evaluation Results ===")
        print(f"Total assets: {summary['overall']['total_assets']}")
        print(f"Successful evaluations: {summary['overall']['successful_evaluations']}")
        print(f"Failed evaluations: {summary['overall']['failed_evaluations']}")
        print(f"Mean FD_dinov2: {summary['overall']['mean_fd_dinov2']:.4f}")
        print(f"Std FD_dinov2: {summary['overall']['std_fd_dinov2']:.4f}")
        print(f"Median FD_dinov2: {summary['overall']['median_fd_dinov2']:.4f}")
        print(f"Min FD_dinov2: {summary['overall']['min_fd_dinov2']:.4f}")
        print(f"Max FD_dinov2: {summary['overall']['max_fd_dinov2']:.4f}")
        
        print("\n=== Results by LLM Model ===")
        for llm_model, stats in summary['by_llm_model'].items():
            print(f"\n{llm_model}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean FD: {stats['mean_fd_dinov2']:.4f}")
            print(f"  Std FD: {stats['std_fd_dinov2']:.4f}")
            print(f"  Median FD: {stats['median_fd_dinov2']:.4f}")
            print(f"  Min FD: {stats['min_fd_dinov2']:.4f}")
            print(f"  Max FD: {stats['max_fd_dinov2']:.4f}")
        
        # Save results
        results_df.to_csv(output_path, index=False)
        
        # Save summary
        summary_path = os.path.splitext(output_path)[0] + '_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        print(f"Summary saved to: {summary_path}")
        if save_renders:
            print(f"Renders and features saved to: {save_dir}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(description='FD_dinov2 Evaluation with Blender Rendering')
    parser.add_argument('--generated_dir', type=str, required=True,
                       help='Directory containing generated 3D assets')
    parser.add_argument('--ground_truth_dir', type=str, required=True,
                       help='Directory containing ground truth 3D assets')
    parser.add_argument('--sampled_csv', type=str, required=True,
                       help='CSV file with sampled asset names')
    parser.add_argument('--results_excel', type=str, required=True,
                       help='Excel file with LLM generation results')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output CSV file path for results')
    parser.add_argument('--save_renders', action='store_true',
                       help='Save rendered images and extracted features')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use for DINOv2 inference')
    parser.add_argument('--max_assets', type=int, default=None,
                       help='Maximum number of assets to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = FD_dinov2_BlenderEvaluator(device=args.device)
    
    # Run evaluation
    results_df = evaluator.evaluate_dataset(
        generated_dir=args.generated_dir,
        ground_truth_dir=args.ground_truth_dir,
        sampled_csv=args.sampled_csv,
        results_excel=args.results_excel,
        output_path=args.output_path,
        save_renders=args.save_renders,
        max_assets=args.max_assets
    )
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()