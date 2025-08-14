#!/usr/bin/env python3
"""
CLIP Score Evaluator for Toys4k dataset - Development Version (No CLIP Model)
This version skips actual CLIP inference and generates mock scores for testing.
"""

import os
import json
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

class Toys4kCLIPEvaluatorNoClip:
    """
    Mock CLIP Score evaluator for development and testing.
    This version generates random scores instead of using actual CLIP inference.
    """
    
    def __init__(self):
        """Initialize the mock evaluator."""
        print("Initializing Mock CLIP Score Evaluator (No actual CLIP model)")
        
        # Rendering parameters for 8 viewpoints
        self.yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 8 viewpoints at 45° intervals
        self.pitch_angle = 30  # Fixed pitch angle
        self.radius = 2  # Fixed radius
        self.fov = 40  # Field of view in degrees
        self.resolution = 512  # Rendering resolution
        
        print(f"Configured for {len(self.yaw_angles)} viewpoints")
    
    def load_toys4k_metadata(self, dataset_path: str) -> pd.DataFrame:
        """Load Toys4k metadata with captions."""
        metadata_path = os.path.join(dataset_path, 'metadata.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        print(f"Loaded metadata for {len(df)} assets")
        
        # Parse captions
        def parse_captions(caption_str):
            try:
                captions = ast.literal_eval(caption_str)
                return captions if isinstance(captions, list) else [caption_str]
            except:
                return [caption_str]
        
        df['parsed_captions'] = df['captions'].apply(parse_captions)
        return df
    
    def find_asset_file(self, dataset_path: str, row: pd.Series) -> Optional[str]:
        """Find the actual asset file (.obj or .blend) for a metadata row."""
        # Try obj files first
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
        """Load a 3D asset from file."""
        try:
            mesh = trimesh.load(asset_path)
            
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
        """Render a 3D asset from 8 different viewpoints using matplotlib."""
        rendered_images = []
        
        # Normalize mesh to unit sphere
        mesh_centered = mesh.copy()
        mesh_centered.vertices -= mesh_centered.centroid
        scale = 1.0 / mesh_centered.bounds.ptp().max() if mesh_centered.bounds.ptp().max() > 0 else 1.0
        mesh_centered.vertices *= scale
        
        try:
            for yaw in self.yaw_angles:
                # Create matplotlib figure
                fig = plt.figure(figsize=(6, 6), dpi=64)
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
    
    def calculate_mock_clip_score(self, num_images: int, caption: str) -> float:
        """Generate a mock CLIP score based on simple heuristics."""
        # Generate score based on caption length and number of images
        base_score = 0.15 + (len(caption.split()) * 0.01)
        noise = np.random.normal(0, 0.05)  # Add some noise
        score = np.clip(base_score + noise, 0.0, 1.0)
        
        # Slightly higher scores for more detailed captions
        if len(caption.split()) > 10:
            score += 0.05
        
        return score
    
    def evaluate_single_asset(self, dataset_path: str, row: pd.Series) -> Dict:
        """Evaluate mock CLIP score for a single 3D asset."""
        result = {
            'sha256': row['sha256'],
            'file_identifier': row['file_identifier'],
            'aesthetic_score': row.get('aesthetic_score', 0.0),
            'clip_score': 0.0,
            'num_views_rendered': 0,
            'success': False,
            'caption_used': '',
            'error': '',
            'mock_evaluation': True
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
            
            # Get the first caption
            captions = row['parsed_captions']
            if not captions:
                result['error'] = 'No captions available'
                return result
            
            caption = captions[0]
            result['caption_used'] = caption
            
            # Render from multiple viewpoints
            rendered_images = self.render_asset_multiview(mesh)
            if not rendered_images:
                result['error'] = 'Rendering failed'
                return result
            
            result['num_views_rendered'] = len(rendered_images)
            
            # Generate mock CLIP score
            clip_score = self.calculate_mock_clip_score(len(rendered_images), caption)
            
            result['clip_score'] = clip_score
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def evaluate_toys4k_dataset(self, dataset_path: str, output_path: str = None, max_assets: int = None) -> Dict:
        """Evaluate mock CLIP scores for the Toys4k dataset."""
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
            
            # Print progress every 5 assets for small tests
            if (idx + 1) % 5 == 0:
                current_success_rate = successful_evaluations / (idx + 1)
                current_avg_score = total_clip_score / successful_evaluations if successful_evaluations > 0 else 0
                print(f"Progress: {idx + 1}/{len(metadata_df)}, "
                      f"Success rate: {current_success_rate:.2%}, "
                      f"Avg mock CLIP score: {current_avg_score:.4f}")
        
        # Calculate aggregated metrics
        mean_clip_score = total_clip_score / successful_evaluations if successful_evaluations > 0 else 0.0
        mean_clip_score_scaled = mean_clip_score * 100
        
        aggregated_results = {
            'mean_clip_score': mean_clip_score,
            'mean_clip_score_scaled': mean_clip_score_scaled,
            'total_assets': len(metadata_df),
            'successful_evaluations': successful_evaluations,
            'success_rate': successful_evaluations / len(metadata_df) if len(metadata_df) > 0 else 0.0,
            'dataset_path': dataset_path,
            'mock_evaluation': True
        }
        
        print(f"\n=== Toys4k Mock CLIP Score Evaluation Results ===")
        print(f"Dataset: {dataset_path}")
        print(f"Total assets: {len(metadata_df)}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Success rate: {aggregated_results['success_rate']:.2%}")
        print(f"Mean Mock CLIP Score: {mean_clip_score:.4f}")
        print(f"Mean Mock CLIP Score (×100): {mean_clip_score_scaled:.2f}")
        print("Note: These are mock scores for testing. Use the full evaluator for real results.")
        
        # Save detailed results to CSV
        if output_path:
            df_results = pd.DataFrame(results)
            df_results.to_csv(output_path, index=False)
            print(f"Mock results saved to: {output_path}")
            
            # Also save summary
            summary_path = output_path.replace('.csv', '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(aggregated_results, f, indent=2)
            print(f"Summary saved to: {summary_path}")
        
        return aggregated_results


def main():
    parser = argparse.ArgumentParser(description='Mock CLIP Score Evaluation for Toys4k Dataset')
    parser.add_argument('--dataset_path', type=str, default='/mnt/nas/Benchmark_Datatset/Toys4k',
                        help='Path to the Toys4k dataset directory')
    parser.add_argument('--output_path', type=str, default='toys4k_mock_clip_scores.csv',
                        help='Path to save detailed results CSV file')
    parser.add_argument('--max_assets', type=int, default=10,
                        help='Maximum number of assets to evaluate (default: 10 for testing)')
    
    args = parser.parse_args()
    
    # Initialize mock evaluator
    evaluator = Toys4kCLIPEvaluatorNoClip()
    
    # Run evaluation
    results = evaluator.evaluate_toys4k_dataset(
        args.dataset_path, args.output_path, args.max_assets
    )
    
    print(f"\nFinal Mock CLIP Score (×100): {results['mean_clip_score_scaled']:.2f}")
    print("\nThis was a mock evaluation. For real CLIP scores, use toys4k_clip_evaluator.py")


if __name__ == "__main__":
    main()