import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path

# CLIP imports
from transformers import CLIPProcessor, CLIPModel

import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CLIPScoreEvaluator:
    """
    CLIP Score evaluator for TRELLIS 3D generation results using Toys4k dataset.
    
    This evaluator:
    1. Renders 3D assets from 8 viewpoints (45° intervals, pitch=30°, radius=2, FoV=40°)
    2. Extracts CLIP features from rendered images and text prompts
    3. Calculates cosine similarity between image and text features
    4. Aggregates scores across views and assets
    """
    
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP Score evaluator.
        
        Args:
            clip_model_name: Name of the CLIP model to use
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model and processor
        print(f"Loading CLIP model: {clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()
        
        # Rendering parameters for 8 viewpoints
        self.yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 viewpoints at 45° intervals
        self.pitch_angle = 30  # Fixed pitch angle
        self.radius = 2  # Fixed radius
        self.fov = 40  # Field of view in degrees
        self.resolution = 512  # Rendering resolution
        
        print(f"Initialized CLIP Score evaluator with {len(self.yaw_angles)} viewpoints")
    
    def load_3d_asset(self, asset_path: str) -> Optional[trimesh.Trimesh]:
        """
        Load a 3D asset from file (.glb, .ply, .obj, etc.).
        
        Args:
            asset_path: Path to the 3D asset file
            
        Returns:
            Trimesh object or None if loading fails
        """
        try:
            if not os.path.exists(asset_path):
                print(f"Asset file not found: {asset_path}")
                return None
            
            # Load mesh using trimesh
            mesh = trimesh.load(asset_path)
            
            # Handle scene objects
            if isinstance(mesh, trimesh.Scene):
                # Get the first mesh in the scene
                meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                if meshes:
                    mesh = meshes[0]
                else:
                    print(f"No valid mesh found in scene: {asset_path}")
                    return None
            
            if not isinstance(mesh, trimesh.Trimesh):
                print(f"Invalid mesh format: {asset_path}")
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
        scale = 1.0 / mesh_centered.bounds.ptp().max()
        mesh_centered.vertices *= scale
        
        try:
            for yaw in self.yaw_angles:
                # Calculate camera position
                yaw_rad = np.deg2rad(yaw)
                pitch_rad = np.deg2rad(self.pitch_angle)
                
                # Camera position in spherical coordinates
                x = self.radius * np.sin(yaw_rad) * np.cos(pitch_rad)
                y = self.radius * np.cos(yaw_rad) * np.cos(pitch_rad)
                z = self.radius * np.sin(pitch_rad)
                
                # Create matplotlib figure
                fig = plt.figure(figsize=(8, 8), dpi=64)
                ax = fig.add_subplot(111, projection='3d')
                
                # Set camera position
                ax.view_init(elev=self.pitch_angle, azim=yaw)
                
                # Render mesh
                vertices = mesh_centered.vertices
                faces = mesh_centered.faces
                
                # Create 3D polygon collection
                poly3d = [[vertices[face] for face in faces]]
                mesh_collection = Poly3DCollection(poly3d, alpha=0.8, facecolor='lightblue', edgecolor='gray')
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
                img_buffer = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_buffer = img_buffer.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                # Resize to target resolution
                from PIL import Image
                img_pil = Image.fromarray(img_buffer)
                img_pil = img_pil.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
                img_array = np.array(img_pil)
                
                rendered_images.append(img_array)
                
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
    
    def evaluate_single_asset(self, asset_path: str, prompt: str) -> Dict:
        """
        Evaluate CLIP score for a single 3D asset.
        
        Args:
            asset_path: Path to the 3D asset file
            prompt: Text prompt used to generate the asset
            
        Returns:
            Dictionary with evaluation results
        """
        result = {
            'asset_path': asset_path,
            'prompt': prompt,
            'clip_score': 0.0,
            'num_views_rendered': 0,
            'success': False
        }
        
        # Load 3D asset
        mesh = self.load_3d_asset(asset_path)
        if mesh is None:
            return result
        
        # Render from multiple viewpoints
        rendered_images = self.render_asset_multiview(mesh)
        if not rendered_images:
            return result
        
        result['num_views_rendered'] = len(rendered_images)
        
        # Extract features
        image_features = self.extract_image_features(rendered_images)
        text_features = self.extract_text_features([prompt])
        
        # Calculate CLIP score
        clip_score = self.calculate_clip_score(image_features, text_features)
        
        result['clip_score'] = clip_score
        result['success'] = True
        
        return result
    
    def evaluate_dataset(self, dataset_path: str, output_path: str = None) -> Dict:
        """
        Evaluate CLIP scores for the entire Toys4k dataset.
        
        Args:
            dataset_path: Path to the Toys4k dataset directory
            output_path: Path to save results CSV file
            
        Returns:
            Dictionary with aggregated results
        """
        # Find all 3D asset files in the dataset
        asset_extensions = ['.glb', '.ply', '.obj', '.gltf']
        asset_files = []
        
        for ext in asset_extensions:
            asset_files.extend(Path(dataset_path).glob(f'**/*{ext}'))
        
        print(f"Found {len(asset_files)} 3D assets in {dataset_path}")
        
        if len(asset_files) == 0:
            print("No 3D assets found!")
            return {'mean_clip_score': 0.0, 'num_assets': 0}
        
        # Evaluate each asset
        results = []
        successful_evaluations = 0
        total_clip_score = 0.0
        
        for asset_file in tqdm(asset_files, desc="Evaluating assets"):
            # For this implementation, we'll use the filename as prompt
            # In practice, you would load the actual prompts from metadata
            prompt = asset_file.stem.replace('_', ' ').replace('-', ' ')
            
            result = self.evaluate_single_asset(str(asset_file), prompt)
            results.append(result)
            
            if result['success']:
                successful_evaluations += 1
                total_clip_score += result['clip_score']
        
        # Calculate aggregated metrics
        mean_clip_score = total_clip_score / successful_evaluations if successful_evaluations > 0 else 0.0
        mean_clip_score_scaled = mean_clip_score * 100  # Scale by 100 as mentioned in instructions
        
        aggregated_results = {
            'mean_clip_score': mean_clip_score,
            'mean_clip_score_scaled': mean_clip_score_scaled,
            'num_assets': len(asset_files),
            'successful_evaluations': successful_evaluations,
            'success_rate': successful_evaluations / len(asset_files) if len(asset_files) > 0 else 0.0
        }
        
        print(f"\n=== CLIP Score Evaluation Results ===")
        print(f"Total assets: {len(asset_files)}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Success rate: {aggregated_results['success_rate']:.2%}")
        print(f"Mean CLIP Score: {mean_clip_score:.4f}")
        print(f"Mean CLIP Score (×100): {mean_clip_score_scaled:.2f}")
        
        # Save detailed results to CSV
        if output_path:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"Detailed results saved to: {output_path}")
        
        return aggregated_results
    
    def evaluate_with_prompts_file(self, assets_dir: str, prompts_file: str, output_path: str = None) -> Dict:
        """
        Evaluate CLIP scores using a separate prompts file.
        
        Args:
            assets_dir: Directory containing 3D assets
            prompts_file: JSON/CSV file containing asset-prompt mappings
            output_path: Path to save results CSV file
            
        Returns:
            Dictionary with aggregated results
        """
        # Load prompts from file
        if prompts_file.endswith('.json'):
            with open(prompts_file, 'r') as f:
                prompts_data = json.load(f)
        elif prompts_file.endswith('.csv'):
            prompts_data = pd.read_csv(prompts_file).to_dict('records')
        else:
            raise ValueError("Prompts file must be JSON or CSV format")
        
        results = []
        successful_evaluations = 0
        total_clip_score = 0.0
        
        for item in tqdm(prompts_data, desc="Evaluating assets with prompts"):
            if isinstance(item, dict):
                asset_name = item.get('asset_name') or item.get('filename')
                prompt = item.get('prompt') or item.get('text')
            else:
                # Handle different data structures
                asset_name = str(item)
                prompt = str(item)
            
            # Find asset file
            asset_path = None
            for ext in ['.glb', '.ply', '.obj', '.gltf']:
                candidate_path = os.path.join(assets_dir, f"{asset_name}{ext}")
                if os.path.exists(candidate_path):
                    asset_path = candidate_path
                    break
            
            if asset_path is None:
                print(f"Asset not found: {asset_name}")
                continue
            
            result = self.evaluate_single_asset(asset_path, prompt)
            results.append(result)
            
            if result['success']:
                successful_evaluations += 1
                total_clip_score += result['clip_score']
        
        # Calculate aggregated metrics
        mean_clip_score = total_clip_score / successful_evaluations if successful_evaluations > 0 else 0.0
        mean_clip_score_scaled = mean_clip_score * 100
        
        aggregated_results = {
            'mean_clip_score': mean_clip_score,
            'mean_clip_score_scaled': mean_clip_score_scaled,
            'num_assets': len(prompts_data),
            'successful_evaluations': successful_evaluations,
            'success_rate': successful_evaluations / len(prompts_data) if len(prompts_data) > 0 else 0.0
        }
        
        print(f"\n=== CLIP Score Evaluation Results ===")
        print(f"Total assets: {len(prompts_data)}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Success rate: {aggregated_results['success_rate']:.2%}")
        print(f"Mean CLIP Score: {mean_clip_score:.4f}")
        print(f"Mean CLIP Score (×100): {mean_clip_score_scaled:.2f}")
        
        # Save detailed results
        if output_path:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            print(f"Detailed results saved to: {output_path}")
        
        return aggregated_results


def main():
    parser = argparse.ArgumentParser(description='CLIP Score Evaluation for TRELLIS 3D Generation')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the Toys4k dataset directory')
    parser.add_argument('--prompts_file', type=str, default=None,
                        help='Optional JSON/CSV file with asset-prompt mappings')
    parser.add_argument('--output_path', type=str, default='clip_score_results.csv',
                        help='Path to save detailed results CSV file')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                        help='CLIP model to use for evaluation')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CLIPScoreEvaluator(clip_model_name=args.clip_model)
    
    # Run evaluation
    if args.prompts_file:
        results = evaluator.evaluate_with_prompts_file(
            args.dataset_path, args.prompts_file, args.output_path
        )
    else:
        results = evaluator.evaluate_dataset(args.dataset_path, args.output_path)
    
    # Save aggregated results
    summary_path = args.output_path.replace('.csv', '_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Summary results saved to: {summary_path}")


if __name__ == "__main__":
    main()