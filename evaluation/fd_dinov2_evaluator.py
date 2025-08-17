#!/usr/bin/env python3
"""
FD_dinov2 Evaluator for TRELLIS Generated 3D Assets

This evaluator measures the Fréchet Distance between generated and ground truth 
3D assets using DINOv2 features extracted from 4-view renderings.
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

class FDDinoV2Evaluator:
    """
    FD_dinov2 evaluator for TRELLIS-generated 3D assets.
    
    This evaluator:
    1. Renders both generated and ground truth assets from 4 viewpoints
    2. Extracts DINOv2 features from rendered images
    3. Calculates Fréchet Distance between feature distributions
    4. Lower FD values indicate better visual quality
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
        
        # Rendering parameters for 4 viewpoints (as specified in TRELLIS paper)
        self.yaw_angles = [0, 90, 180, 270]  # 4 viewpoints at 90° intervals
        self.pitch_angle = 30  # Fixed pitch angle
        self.radius = 2  # Fixed radius
        self.fov = 40  # Field of view in degrees
        self.resolution = 518  # DINOv2 expects 518x518 images
        
        # Image preprocessing for DINOv2
        self.transform = transforms.Compose([
            transforms.Resize((518, 518), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Initialized FD_dinov2 evaluator with {len(self.yaw_angles)} viewpoints on {self.device}")
    
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
    
    def find_asset_file(self, dataset_path: str, file_identifier: str) -> Optional[str]:
        """
        Find the actual asset file (.obj or .blend) for a file identifier.
        
        Args:
            dataset_path: Path to Toys4k dataset
            file_identifier: File identifier (e.g., 'giraffe/giraffe_006/giraffe_006.blend')
            
        Returns:
            Path to asset file or None
        """
        # Try obj files first (easier to load with trimesh)
        obj_path = os.path.join(dataset_path, 'toys4k_obj_files')
        parts = file_identifier.split('/')
        if len(parts) >= 2:
            category = parts[0]
            asset_name = parts[1]
            obj_file = os.path.join(obj_path, category, asset_name, 'mesh.obj')
            if os.path.exists(obj_file):
                return obj_file
        
        # Fallback: try blend files
        blend_path = os.path.join(dataset_path, 'toys4k_blend_files')
        blend_file = os.path.join(blend_path, file_identifier)
        if os.path.exists(blend_file):
            return blend_file
        
        return None
    
    def find_generated_assets(self, output_base_path: str, llm_model: str, object_name: str) -> List[str]:
        """
        Find generated 3D assets (.glb/.ply) for a specific LLM model and object.
        
        Args:
            output_base_path: Base path to generated outputs
            llm_model: LLM model name (category from results)
            object_name: Clean object name
            
        Returns:
            List of asset file paths
        """
        # Convert llm_model category to directory name
        model_dir_map = {
            'gemma3': 'gemma3_1b',
            'qwen3': 'qwen3_0.6b',
            'deepseek': 'deepseek-r1_1.5b',
            'gpt': 'gpt-oss_20b',
            'llama': 'llama3.1_8b'
        }
        
        model_dir = model_dir_map.get(llm_model, llm_model)
        asset_dir = os.path.join(output_base_path, model_dir, object_name)
        
        if not os.path.exists(asset_dir):
            return []
        
        # Find .glb and .ply files
        import glob
        glb_files = glob.glob(os.path.join(asset_dir, "*.glb"))
        ply_files = glob.glob(os.path.join(asset_dir, "*.ply"))
        
        return glb_files + ply_files
    
    def load_3d_asset(self, asset_path: str) -> Optional[trimesh.Trimesh]:
        """
        Load a 3D asset from file.
        
        Args:
            asset_path: Path to the 3D asset file
            
        Returns:
            Trimesh object or None if loading fails
        """
        try:
            mesh = trimesh.load(asset_path)
            
            # Handle scene objects
            if isinstance(mesh, trimesh.Scene):
                meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                if meshes:
                    mesh = meshes[0]
                else:
                    return None
            
            if not isinstance(mesh, trimesh.Trimesh):
                return None
                
            return mesh
                
        except Exception as e:
            print(f"Error loading asset {asset_path}: {e}")
            return None
    
    def render_asset_4views(self, mesh: trimesh.Trimesh) -> List[np.ndarray]:
        """
        Render a 3D asset from 4 different viewpoints using matplotlib.
        
        Args:
            mesh: 3D mesh to render
            
        Returns:
            List of rendered images as numpy arrays
        """
        rendered_images = []
        
        # Normalize mesh to unit sphere
        mesh_centered = mesh.copy()
        mesh_centered.vertices -= mesh_centered.centroid
        scale = 1.0 / mesh_centered.bounds.ptp().max() if mesh_centered.bounds.ptp().max() > 0 else 1.0
        mesh_centered.vertices *= scale
        
        try:
            for yaw in self.yaw_angles:
                # Create matplotlib figure
                fig = plt.figure(figsize=(8, 8), dpi=65)  # Slightly higher DPI for 518x518
                ax = fig.add_subplot(111, projection='3d')
                
                # Set camera position
                ax.view_init(elev=self.pitch_angle, azim=yaw)
                
                # Render mesh
                vertices = mesh_centered.vertices
                faces = mesh_centered.faces
                
                # Create 3D polygon collection
                face_vertices = vertices[faces]
                mesh_collection = Poly3DCollection(face_vertices, alpha=0.8, facecolor='lightblue', edgecolor='gray')
                ax.add_collection3d(mesh_collection)
                
                # Set equal aspect ratio and limits
                max_range = 1.2
                ax.set_xlim([-max_range, max_range])
                ax.set_ylim([-max_range, max_range])
                ax.set_zlim([-max_range, max_range])
                
                # Remove axes and background
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.grid(False)
                ax.set_facecolor('white')
                
                # Convert to image
                fig.canvas.draw()
                # Use buffer_rgba and convert to rgb
                buf = fig.canvas.buffer_rgba()
                img_array = np.asarray(buf)
                img_rgb = img_array[:, :, :3]  # Remove alpha channel
                
                # Resize to target resolution (518x518 for DINOv2)
                img_pil = Image.fromarray(img_rgb.astype(np.uint8))
                img_pil = img_pil.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
                img_final = np.array(img_pil)
                
                rendered_images.append(img_final)
                plt.close(fig)
                
        except Exception as e:
            print(f"Error rendering asset: {e}")
            return []
        
        return rendered_images
    
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
                                  dataset_path: str, 
                                  output_base_path: str,
                                  sampled_row: pd.Series, 
                                  llm_row: pd.Series) -> Dict:
        """
        Evaluate FD_dinov2 for a single asset pair (ground truth vs generated).
        
        Args:
            dataset_path: Path to Toys4k dataset
            output_base_path: Path to generated assets
            sampled_row: Row from sampled data
            llm_row: Row from LLM results
            
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
            'error': ''
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
                          llm_models_filter: List[str] = None) -> Dict:
        """
        Evaluate FD_dinov2 scores for TRELLIS-generated assets vs ground truth.
        
        Args:
            sampled_csv_path: Path to sampled_data_100_random.csv
            results_excel_path: Path to Excel file with LLM results
            dataset_path: Path to Toys4k dataset
            output_base_path: Base path to generated outputs directory
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
        
        for i, (sampled_row, llm_row) in enumerate(tqdm(matched_pairs, desc="Evaluating FD_dinov2")):
            result = self.evaluate_single_asset_pair(dataset_path, output_base_path, sampled_row, llm_row)
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
            
            # Print progress every 10 assets
            if (i + 1) % 10 == 0:
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
        description='FD_dinov2 Evaluation for TRELLIS Generated 3D Assets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 5 asset pairs
  python fd_dinov2_evaluator.py --max_assets 5 --save_path test_fd_results.csv
  
  # Evaluate specific LLM models
  python fd_dinov2_evaluator.py \\
    --llm_models gemma3 qwen3 \\
    --save_path model_fd_results.csv
  
  # Full evaluation
  python fd_dinov2_evaluator.py --save_path full_fd_dinov2_results.csv
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
                        default='fd_dinov2_results.csv',
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
    
    print(f"📊 FD_dinov2 Evaluation for TRELLIS Generated Assets")
    print(f"📁 Sampled CSV: {args.sampled_csv}")
    print(f"📁 LLM results: {args.results_excel}")
    print(f"📁 Dataset path: {args.dataset_path}")
    print(f"📁 Generated assets: {args.output_base_path}")
    print(f"💾 Results will be saved to: {args.save_path}")
    if args.llm_models:
        print(f"🔍 Evaluating models: {', '.join(args.llm_models)}")
    if args.max_assets:
        print(f"⚡ Limited to {args.max_assets} asset pairs for testing")
    print()
    
    # Initialize evaluator
    evaluator = FDDinoV2Evaluator(device=args.device)
    
    # Run evaluation
    results = evaluator.evaluate_fd_dinov2(
        args.sampled_csv, args.results_excel, args.dataset_path, 
        args.output_base_path, args.save_path, args.max_assets, args.llm_models
    )
    
    print(f"\n🎯 Final FD_dinov2: {results['mean_fd_dinov2']:.4f}")
    print(f"   (Lower values indicate better visual quality)")
    print(f"✅ Results saved to: {args.save_path}")


if __name__ == "__main__":
    main()