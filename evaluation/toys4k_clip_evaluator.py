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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import trimesh

# CLIP imports
from transformers import CLIPProcessor, CLIPModel

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
        self.pitch_angle = 30  # Fixed pitch angle
        self.radius = 2  # Fixed radius
        self.fov = 40  # Field of view in degrees
        self.resolution = 512  # Rendering resolution
        
        print(f"Initialized CLIP Score evaluator with {len(self.yaw_angles)} viewpoints")
    
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
    
    def find_asset_file(self, dataset_path: str, row: pd.Series) -> Optional[str]:
        """
        Find the actual asset file (.obj or .blend) for a metadata row.
        
        Args:
            dataset_path: Path to Toys4k dataset
            row: Metadata row
            
        Returns:
            Path to asset file or None
        """
        # Try obj files first (easier to load with trimesh)
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
        
        # Fallback: try blend files (but trimesh might not support them)
        if pd.notna(row.get('local_path', '')):
            blend_file = row['local_path'].replace('/mnt/sdc_870evo_8TB/Toys4k/', f'{dataset_path}/')
            if os.path.exists(blend_file):
                return blend_file
        
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
    
    def render_asset_multiview(self, mesh: trimesh.Trimesh) -> List[np.ndarray]:
        """
        Render a 3D asset from 8 different viewpoints using matplotlib.
        
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
                fig = plt.figure(figsize=(8, 8), dpi=64)
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
                
                # Resize to target resolution
                img_pil = Image.fromarray(img_rgb.astype(np.uint8))
                img_pil = img_pil.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
                img_final = np.array(img_pil)
                
                rendered_images.append(img_final)
                plt.close(fig)
                
        except Exception as e:
            print(f"Error rendering asset: {e}")
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
    
    def evaluate_single_asset(self, dataset_path: str, row: pd.Series) -> Dict:
        """
        Evaluate CLIP score for a single 3D asset from Toys4k.
        
        Args:
            dataset_path: Path to Toys4k dataset
            row: Metadata row for the asset
            
        Returns:
            Dictionary with evaluation results
        """
        result = {
            'sha256': row['sha256'],
            'file_identifier': row['file_identifier'],
            'aesthetic_score': row.get('aesthetic_score', 0.0),
            'clip_score': 0.0,
            'num_views_rendered': 0,
            'success': False,
            'caption_used': '',
            'error': ''
        }
        
        try:
            # Find asset file
            asset_path = self.find_asset_file(dataset_path, row)
            if asset_path is None:
                result['error'] = 'Asset file not found'
                return result
            
            # Load 3D asset
            mesh = self.load_3d_asset(asset_path)
            if mesh is None:
                result['error'] = 'Failed to load mesh'
                return result
            
            # Get the first (most detailed) caption
            captions = row['parsed_captions']
            if not captions:
                result['error'] = 'No captions available'
                return result
            
            caption = captions[0]  # Use the most detailed caption
            result['caption_used'] = caption
            
            # Render from multiple viewpoints
            rendered_images = self.render_asset_multiview(mesh)
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
    
    def evaluate_toys4k_dataset(self, dataset_path: str, output_path: str = None, max_assets: int = None) -> Dict:
        """
        Evaluate CLIP scores for the Toys4k dataset.
        
        Args:
            dataset_path: Path to Toys4k dataset directory
            output_path: Path to save results CSV file
            max_assets: Maximum number of assets to evaluate (for testing)
            
        Returns:
            Dictionary with aggregated results
        """
        # Load metadata
        metadata_df = self.load_toys4k_metadata(dataset_path)
        
        if max_assets:
            metadata_df = metadata_df.head(max_assets)
            print(f"Limiting evaluation to first {max_assets} assets")
        
        # Evaluate each asset
        results = []
        successful_evaluations = 0
        total_clip_score = 0.0
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Evaluating assets"):
            result = self.evaluate_single_asset(dataset_path, row)
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
            'dataset_path': dataset_path
        }
        
        print(f"\n=== Toys4k CLIP Score Evaluation Results ===")
        print(f"Dataset: {dataset_path}")
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
        args.dataset_path, args.output_path, args.max_assets
    )
    
    print(f"\nFinal CLIP Score (×100): {results['mean_clip_score_scaled']:.2f}")


if __name__ == "__main__":
    main()