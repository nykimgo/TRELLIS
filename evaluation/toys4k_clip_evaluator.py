#!/usr/bin/env python3
"""
CLIP Score Evaluator specifically designed for Toys4k dataset
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from PIL import Image
from tqdm import tqdm
import argparse
import ast
import trimesh
import tempfile
import shutil
import glob
import subprocess
from subprocess import DEVNULL, call

# CLIP imports
from transformers import CLIPProcessor, CLIPModel

# Blender rendering setup
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

class Toys4kCLIPEvaluator:
    """
    CLIP Score evaluator specifically for Toys4k dataset.
    
    This evaluator:
    1. Loads Toys4k metadata with captions
    2. Renders 3D assets from 8 viewpoints (45° intervals, pitch=30°, radius=2, FoV=40°)
    3. Extracts CLIP features from rendered images and captions
    4. Calculates cosine similarity between image and text features
    5. Aggregates scores across views and assets
    """
    
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP Score evaluator for Toys4k.
        
        Args:
            clip_model_name: Name of the CLIP model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model and processor with safetensors
        print(f"Loading CLIP model: {clip_model_name}")
        try:
            # Try to load with safetensors first (recommended for security)
            self.clip_model = CLIPModel.from_pretrained(
                clip_model_name, 
                use_safetensors=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            print("✓ Loaded model with safetensors")
        except Exception as e:
            print(f"Failed to load with safetensors: {e}")
            print("Trying alternative CLIP model...")
            # Try a different model that supports safetensors
            try:
                clip_model_name = "openai/clip-vit-base-patch16"
                self.clip_model = CLIPModel.from_pretrained(
                    clip_model_name,
                    use_safetensors=True,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).to(self.device)
                self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
                print(f"✓ Loaded alternative model: {clip_model_name}")
            except Exception as e2:
                print(f"Failed to load alternative model: {e2}")
                raise RuntimeError(
                    "Could not load CLIP model due to torch version compatibility. "
                    "Please upgrade torch to version >=2.6 or use a model with safetensors support."
                )
        
        self.clip_model.eval()
        
        # Rendering parameters for 8 viewpoints
        self.yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 viewpoints at 45° intervals
        self.pitch_angle = 30  # Fixed pitch angle in degrees
        self.radius = 2  # Fixed radius
        self.fov = 40  # Field of view in degrees
        self.resolution = 512  # Rendering resolution
        
        # Install Blender if needed
        self._install_blender()
        
        print(f"Initialized CLIP Score evaluator with {len(self.yaw_angles)} viewpoints")
    
    def _install_blender(self):
        """Install Blender if not already available."""
        if not os.path.exists(BLENDER_PATH):
            print("Installing Blender...")
            os.system('sudo apt-get update')
            os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
            os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
            os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')
            print("✓ Blender installed")
        else:
            print("✓ Blender already available")
    
    def load_toys4k_metadata(self, dataset_path: str) -> pd.DataFrame:
        """
        Load Toys4k metadata with captions.
        
        Args:
            dataset_path: Path to Toys4k dataset directory
            
        Returns:
            DataFrame with metadata
        """
        metadata_path = os.path.join(dataset_path, 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        print(f"Loaded metadata for {len(df)} assets")
        
        # Parse captions (they are stored as string representations of lists)
        def parse_captions(caption_str):
            try:
                captions = ast.literal_eval(caption_str)
                return captions if isinstance(captions, list) else [caption_str]
            except:
                return [caption_str]
        
        df['parsed_captions'] = df['captions'].apply(parse_captions)
        return df
    
    def load_excel_data(self, excel_path: str) -> pd.DataFrame:
        """
        Load data from Excel or CSV file containing object_name and sha256.
        
        Args:
            excel_path: Path to Excel (.xlsx/.xls) or CSV file
            
        Returns:
            DataFrame with data
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Data file not found: {excel_path}")
        
        # Check file extension and read accordingly
        if excel_path.endswith('.xlsx') or excel_path.endswith('.xls'):
            df = pd.read_excel(excel_path)
        else:
            df = pd.read_csv(excel_path)
            
        print(f"Loaded {len(df)} entries from {excel_path}")
        
        # Check available columns
        print(f"Available columns: {list(df.columns)}")
        
        # Try to find object_name column (might have different name)
        object_name_col = None
        for col in df.columns:
            if 'object' in col.lower() and 'name' in col.lower():
                object_name_col = col
                break
        
        if object_name_col is None:
            # If no object_name column, use file_identifier or create a default
            if 'file_identifier' in df.columns:
                df['object_name'] = df['file_identifier'].apply(lambda x: x.split('/')[-1] if isinstance(x, str) else 'unknown')
                print("Created object_name from file_identifier")
            else:
                df['object_name'] = 'unknown'
                print("No object_name found, using 'unknown' as default")
        else:
            df['object_name'] = df[object_name_col]
            print(f"Using {object_name_col} as object_name")
        
        # Validate required columns
        if 'sha256' not in df.columns:
            raise ValueError(f"Missing required column 'sha256' in file: {excel_path}")
        
        return df
    
    def find_asset_file(self, dataset_path: str, row: pd.Series) -> Optional[str]:
        """
        Find the actual asset file (.blend or .obj) for a metadata row.
        Prioritize .blend files for color rendering.
        
        Args:
            dataset_path: Path to Toys4k dataset
            row: Metadata row
            
        Returns:
            Path to asset file or None
        """
        # Try blend files first (for color rendering)
        if pd.notna(row.get('local_path', '')):
            blend_file = row['local_path'].replace('/mnt/sdc_870evo_8TB/Toys4k/', f'{dataset_path}/')
            if os.path.exists(blend_file):
                return blend_file
        
        # Fallback: try obj files (but these will render in grayscale)
        obj_path = os.path.join(dataset_path, 'toys4k_obj_files')
        if 'file_identifier' in row:
            # Parse file_identifier to get category and asset name
            parts = row['file_identifier'].split('/')
            if len(parts) >= 2:
                category = parts[0]
                asset_name = parts[1]
                obj_file = os.path.join(obj_path, category, asset_name, 'mesh.obj')
                if os.path.exists(obj_file):
                    return obj_file
        
        return None
    
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
    
    def render_asset_multiview(self, asset_path: str, ground_truth_dir: str, object_name: str, sha256: str) -> List[np.ndarray]:
        """
        Render a 3D asset from 8 different viewpoints using Blender (copying fd_dinov2_blender_evaluator.py approach).
        
        Args:
            asset_path: Path to the 3D asset file
            ground_truth_dir: Path to the Toys4k dataset directory
            object_name: Object name (commas will be removed)
            sha256: SHA256 hash of the asset
            
        Returns:
            List of rendered images as numpy arrays
        """
        rendered_images = []
        
        try:
            # Clean object name by removing commas
            clean_object_name = object_name.replace(',', '') if object_name else 'unknown'
            
            # Use only first 6 characters of sha256
            sha256_short = sha256[:6] if sha256 else 'unknown'
            
            # Create output directory
            output_dir = os.path.join(ground_truth_dir, 'render_multiviews_for_CLIPeval', f'{clean_object_name}_{sha256_short}')
            os.makedirs(output_dir, exist_ok=True)
            
            # Define 8 views: exactly like fd_dinov2_blender_evaluator.py but with 8 angles
            views = []
            for yaw in self.yaw_angles:
                views.append({
                    'yaw': yaw * np.pi / 180, 
                    'pitch': self.pitch_angle * np.pi / 180, 
                    'radius': self.radius, 
                    'fov': self.fov * np.pi / 180
                })
            
            # Prepare Blender command (same as dataset_toolkits/render.py)
            args = [
                BLENDER_PATH, '-b', '-P', '/home/sr/TRELLIS/dataset_toolkits/blender_script/render.py',
                '--',
                '--views', json.dumps(views),
                '--object', os.path.expanduser(asset_path),
                '--resolution', '512',
                '--output_folder', output_dir,
                '--engine', 'CYCLES'
            ]
            
            # Handle .blend files (exactly like fd_dinov2_blender_evaluator.py)
            if asset_path.endswith('.blend'):
                args.insert(1, asset_path)
                
            # Run Blender rendering with colorful materials
            try:
                result = subprocess.run(args, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"Blender rendering failed: {result.stderr}")
                    return []
            except subprocess.TimeoutExpired:
                print(f"Blender rendering timed out for {asset_path}")
                return []
                
            # Collect rendered images and rename them to desired format
            for i, yaw in enumerate(self.yaw_angles):
                temp_img_path = os.path.join(output_dir, f'{i:03d}.png')
                final_img_path = os.path.join(output_dir, f'gt_view_{yaw}.png')
                
                if os.path.exists(temp_img_path):
                    # Rename to final format
                    os.rename(temp_img_path, final_img_path)
                    
                    # Load image for return
                    img = Image.open(final_img_path).convert('RGB')
                    img_array = np.array(img)
                    rendered_images.append(img_array)
                    
        except Exception as e:
            print(f"Error rendering asset with Blender: {e}")
            return []
        
        return rendered_images
    
    def extract_image_features(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Extract CLIP features from rendered images.
        
        Args:
            images: List of rendered images as numpy arrays
            
        Returns:
            Tensor of image features [num_images, feature_dim]
        """
        if not images:
            return torch.empty(0, self.clip_model.config.projection_dim).to(self.device)
        
        # Convert numpy arrays to PIL Images
        pil_images = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)
            pil_images.append(pil_img)
        
        # Process images with CLIP processor
        inputs = self.clip_processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract image features
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            # Normalize features
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        return image_features
    
    def extract_text_features(self, texts: List[str]) -> torch.Tensor:
        """
        Extract CLIP features from text prompts.
        
        Args:
            texts: List of text prompts
            
        Returns:
            Tensor of text features [num_texts, feature_dim]
        """
        if not texts:
            return torch.empty(0, self.clip_model.config.projection_dim).to(self.device)
        
        # Process texts with CLIP processor
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract text features
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            # Normalize features
            text_features = F.normalize(text_features, p=2, dim=-1)
        
        return text_features
    
    def calculate_clip_score(self, image_features: torch.Tensor, text_features: torch.Tensor) -> float:
        """
        Calculate CLIP score as cosine similarity between image and text features.
        
        Args:
            image_features: Image features tensor [num_images, feature_dim]
            text_features: Text features tensor [1, feature_dim]
            
        Returns:
            Average CLIP score across all images
        """
        if image_features.shape[0] == 0 or text_features.shape[0] == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarities = torch.mm(image_features, text_features.t()).squeeze()
        
        # Average across all views
        if similarities.dim() == 0:
            avg_similarity = similarities.item()
        else:
            avg_similarity = similarities.mean().item()
        
        return avg_similarity
    
    def evaluate_single_asset(self, dataset_path: str, metadata_row: pd.Series, excel_row: pd.Series = None) -> Dict:
        """
        Evaluate CLIP score for a single 3D asset from Toys4k.
        
        Args:
            dataset_path: Path to Toys4k dataset
            metadata_row: Metadata row for the asset from metadata.csv
            excel_row: Optional row from Excel file containing object_name
            
        Returns:
            Dictionary with evaluation results
        """
        result = {
            'sha256': metadata_row['sha256'],
            'file_identifier': metadata_row['file_identifier'],
            'aesthetic_score': metadata_row.get('aesthetic_score', 0.0),
            'object_name': excel_row['object_name'] if excel_row is not None else 'unknown',
            'clip_score': 0.0,
            'num_views_rendered': 0,
            'success': False,
            'caption_used': '',
            'error': ''
        }
        
        try:
            # Find asset file
            asset_path = self.find_asset_file(dataset_path, metadata_row)
            if asset_path is None:
                result['error'] = 'Asset file not found'
                return result
            
            # Get the first (most detailed) caption
            captions = metadata_row['parsed_captions']
            if not captions:
                result['error'] = 'No captions available'
                return result
            
            caption = captions[0]  # Use the most detailed caption
            result['caption_used'] = caption
            
            # Get object name and sha256 for directory naming
            object_name = result['object_name']
            sha256 = result['sha256']
            
            # Render from multiple viewpoints using Blender and save to specific directory
            rendered_images = self.render_asset_multiview(asset_path, dataset_path, object_name, sha256)
            if not rendered_images:
                result['error'] = 'Rendering failed'
                return result
            
            result['num_views_rendered'] = len(rendered_images)
            
            # Extract features
            image_features = self.extract_image_features(rendered_images)
            text_features = self.extract_text_features([caption])
            
            # Calculate CLIP score
            clip_score = self.calculate_clip_score(image_features, text_features)
            
            result['clip_score'] = clip_score
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def evaluate_toys4k_dataset(self, dataset_path: str, excel_path: str = None, output_path: str = None, max_assets: int = None) -> Dict:
        """
        Evaluate CLIP scores for the Toys4k dataset.
        
        Args:
            dataset_path: Path to Toys4k dataset directory
            excel_path: Optional path to Excel file with object_name and sha256
            output_path: Path to save results CSV file
            max_assets: Maximum number of assets to evaluate (for testing)
            
        Returns:
            Dictionary with aggregated results
        """
        # Load metadata
        metadata_df = self.load_toys4k_metadata(dataset_path)
        
        # Load Excel data if provided
        excel_df = None
        if excel_path:
            excel_df = self.load_excel_data(excel_path)
            # Filter metadata to only include assets that are in the Excel file
            metadata_df = metadata_df[metadata_df['sha256'].isin(excel_df['sha256'])]
            print(f"Filtered to {len(metadata_df)} assets that match Excel file")
        
        if max_assets:
            metadata_df = metadata_df.head(max_assets)
            print(f"Limiting evaluation to first {max_assets} assets")
        
        # Evaluate each asset
        results = []
        successful_evaluations = 0
        total_clip_score = 0.0
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Evaluating assets"):
            # Find corresponding Excel row if available
            excel_row = None
            if excel_df is not None:
                matching_excel_rows = excel_df[excel_df['sha256'] == row['sha256']]
                if not matching_excel_rows.empty:
                    excel_row = matching_excel_rows.iloc[0]
            
            result = self.evaluate_single_asset(dataset_path, row, excel_row)
            results.append(result)
            
            if result['success']:
                successful_evaluations += 1
                total_clip_score += result['clip_score']
            
            # Print progress every 10 assets
            if (idx + 1) % 10 == 0:
                current_success_rate = successful_evaluations / (idx + 1)
                current_avg_score = total_clip_score / successful_evaluations if successful_evaluations > 0 else 0
                print(f"Progress: {idx + 1}/{len(metadata_df)}, "
                      f"Success rate: {current_success_rate:.2%}, "
                      f"Avg CLIP score: {current_avg_score:.4f}")
        
        # Calculate aggregated metrics
        mean_clip_score = total_clip_score / successful_evaluations if successful_evaluations > 0 else 0.0
        mean_clip_score_scaled = mean_clip_score * 100  # Scale by 100 as mentioned in instructions
        
        aggregated_results = {
            'mean_clip_score': mean_clip_score,
            'mean_clip_score_scaled': mean_clip_score_scaled,
            'total_assets': len(metadata_df),
            'successful_evaluations': successful_evaluations,
            'success_rate': successful_evaluations / len(metadata_df) if len(metadata_df) > 0 else 0.0,
            'dataset_path': dataset_path,
            'excel_path': excel_path
        }
        
        print(f"\n=== Toys4k CLIP Score Evaluation Results ===")
        print(f"Dataset: {dataset_path}")
        if excel_path:
            print(f"Excel file: {excel_path}")
        print(f"Total assets: {len(metadata_df)}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Success rate: {aggregated_results['success_rate']:.2%}")
        print(f"Mean CLIP Score: {mean_clip_score:.4f}")
        print(f"Mean CLIP Score (×100): {mean_clip_score_scaled:.2f}")
        
        # Save detailed results to CSV
        if output_path:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_path, index=False)
            print(f"Detailed results saved to: {output_path}")
            
            # Also save summary
            summary_path = output_path.replace('.csv', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(aggregated_results, f, indent=2)
            print(f"Summary saved to: {summary_path}")
        
        return aggregated_results


def main():
    parser = argparse.ArgumentParser(description='CLIP Score Evaluation for Toys4k Dataset')
    parser.add_argument('--dataset_path', type=str, default='/mnt/nas/Benchmark_Datatset/Toys4k',
                        help='Path to the Toys4k dataset directory')
    parser.add_argument('--results_excel', type=str, default=None,
                        help='Path to Excel file containing object_name and sha256 columns')
    parser.add_argument('--output_path', type=str, default='toys4k_clip_scores.csv',
                        help='Path to save detailed results CSV file')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                        help='CLIP model to use for evaluation')
    parser.add_argument('--max_assets', type=int, default=None,
                        help='Maximum number of assets to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = Toys4kCLIPEvaluator(clip_model_name=args.clip_model)
    
    # Run evaluation
    results = evaluator.evaluate_toys4k_dataset(
        args.dataset_path, args.results_excel, args.output_path, args.max_assets
    )
    
    print(f"\nFinal CLIP Score (×100): {results['mean_clip_score_scaled']:.2f}")


if __name__ == "__main__":
    main()