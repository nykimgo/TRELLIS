#!/usr/bin/env python3
"""
CLIP Score Evaluator for TRELLIS Generated 3D Assets

This evaluator measures CLIP scores between TRELLIS-generated 3D assets and their 
LLM-augmented text prompts, comparing different LLM models' performance.
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
import trimesh
import glob
import tempfile
import shutil
from subprocess import DEVNULL, call

# CLIP imports
from transformers import CLIPProcessor, CLIPModel

# Blender rendering setup
BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

class TrellisGeneratedCLIPEvaluator:
    """
    CLIP Score evaluator for TRELLIS-generated 3D assets.
    
    This evaluator:
    1. Loads LLM-augmented prompts from Excel files
    2. Finds corresponding generated 3D assets (.glb/.ply files)
    3. Renders assets from 8 viewpoints (45° intervals, pitch=30°, radius=2, FoV=40°)
    4. Extracts CLIP features from rendered images and augmented text prompts
    5. Calculates cosine similarity between image and text features
    6. Aggregates scores across views, assets, and LLM models
    """
    
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP Score evaluator for TRELLIS-generated assets.
        
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
    
    def load_llm_results(self, results_path: str) -> pd.DataFrame:
        """
        Load LLM-augmented prompt results from Excel file.
        
        Args:
            results_path: Path to Excel file with LLM results
            
        Returns:
            DataFrame with LLM results
        """
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        df = pd.read_excel(results_path)
        print(f"Loaded {len(df)} LLM-augmented prompts from {results_path}")
        
        # Validate required columns
        required_columns = ['category', 'llm_model', 'object_name_clean', 'user_prompt', 'text_prompt', 'sha256', 'file_identifier']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def find_generated_assets(self, output_base_path: str, llm_model: str, file_identifier: str) -> List[str]:
        """
        Find generated 3D assets (.glb/.ply) for a specific LLM model and object.
        
        Args:
            output_base_path: Base path to generated outputs
            llm_model: LLM model name (category from results)
            object_name: Clean object name
            
        Returns:
            List of asset file paths
        """
        object_name = file_identifier.split('/')[1]
        # Replace ':' with '_' for filesystem path compatibility
        model_dir = llm_model.replace(':', '_')
        
        asset_dir = os.path.join(output_base_path, model_dir, object_name)
        
        if not os.path.exists(asset_dir):
            return []
        
        # Find .glb and .ply files
        glb_files = glob.glob(os.path.join(asset_dir, "*.glb"))
        # ply_files = glob.glob(os.path.join(asset_dir, "*.ply"))

        return glb_files
    
    def load_3d_asset(self, asset_path: str) -> Optional[trimesh.Trimesh]:
        """
        Load a 3D asset from file (.glb or .ply).
        
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
    
    def render_asset_multiview(self, asset_path: str, save_dir: str = None) -> List[np.ndarray]:
        """
        Render a 3D asset from 8 different viewpoints using Blender.
        
        Args:
            asset_path: Path to the 3D asset file
            save_dir: Optional directory to save rendered images
            
        Returns:
            List of rendered images as numpy arrays
        """
        rendered_images = []
        
        try:
            # Use save_dir if provided, otherwise create temporary directory
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                temp_dir = save_dir
                use_temp = False
            else:
                temp_dir = tempfile.mkdtemp()
                use_temp = True
            # Build camera views for 8 viewpoints
            views = []
            for yaw in self.yaw_angles:
                yaw_rad = np.radians(yaw)
                pitch_rad = np.radians(self.pitch_angle)
                fov_rad = np.radians(self.fov)
                
                views.append({
                    'yaw': yaw_rad,
                    'pitch': pitch_rad,
                    'radius': self.radius,
                    'fov': fov_rad
                })
            
            # Blender rendering arguments
            args = [
                BLENDER_PATH, '-b', '-P', 
                os.path.join('/home/sr/TRELLIS/dataset_toolkits/blender_script', 'render.py'),
                '--',
                '--views', json.dumps(views),
                '--object', asset_path,
                '--resolution', str(self.resolution),
                '--output_folder', temp_dir,
                '--engine', 'CYCLES',
            ]
            
            # Run Blender rendering
            call(args, stdout=DEVNULL, stderr=DEVNULL)
            
            # Load rendered images and rename if saving
            for i, yaw in enumerate(self.yaw_angles):
                temp_img_path = os.path.join(temp_dir, f'{i:03d}.png')
                
                if save_dir:
                    # Rename to desired format when saving
                    final_img_path = os.path.join(temp_dir, f'gen_view_{yaw}.png')
                    if os.path.exists(temp_img_path):
                        os.rename(temp_img_path, final_img_path)
                        img_path = final_img_path
                    else:
                        img_path = None
                else:
                    img_path = temp_img_path
                
                if img_path and os.path.exists(img_path):
                    img = Image.open(img_path).convert('RGB')
                    img_array = np.array(img)
                    rendered_images.append(img_array)
                else:
                    print(f"Warning: Rendered image not found: {img_path}")
            
            # Clean up temporary directory if we created it
            if use_temp:
                shutil.rmtree(temp_dir)
            
            if not rendered_images:
                print("No rendered images found - Blender rendering may have failed")
                    
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
    
    def evaluate_single_generated_asset(self, output_base_path: str, row: pd.Series, save_base_path: str = None) -> Dict:
        """
        Evaluate CLIP score for a single TRELLIS-generated 3D asset.
        
        Args:
            output_base_path: Base path to generated outputs
            row: LLM results row
            
        Returns:
            Dictionary with evaluation results
        """
        result = {
            'sha256': row['sha256'],
            'file_identifier': row['file_identifier'],
            'llm_model': row['llm_model'],
            'category': row['category'],
            'object_name_clean': row['object_name_clean'],
            'user_prompt': row['user_prompt'],
            'text_prompt': row['text_prompt'],
            'clip_score': 0.0,
            'num_views_rendered': 0,
            'success': False,
            'asset_path': '',
            'error': ''
        }
        
        try:
            # Find generated asset files
            asset_files = self.find_generated_assets(
                output_base_path, row['llm_model'], row['file_identifier']
            )
            
            if not asset_files:
                result['error'] = f"No generated assets found for {row['llm_model']}/{row['file_identifier']}"
                return result
            
            # Use the first found asset file
            asset_path = asset_files[0]
            result['asset_path'] = asset_path
            
            # Use the LLM-augmented text prompt
            text_prompt = row['text_prompt']
            if pd.isna(text_prompt) or not text_prompt.strip():
                result['error'] = 'No text_prompt available'
                return result
            
            # Prepare save directory for rendered images
            save_dir = None
            if save_base_path:
                # Extract middle part from file_identifier (e.g., keyboard/keyboard_022/keyboard_022.blend -> keyboard_022)
                file_id = row['file_identifier']
                if '/' in file_id:
                    parts = file_id.split('/')
                    if len(parts) >= 2:
                        middle_part = parts[1]  # keyboard_022
                    else:
                        middle_part = parts[0]
                else:
                    middle_part = file_id
                
                # Create save directory: {save_path}/CLIP_evaluation/TRELLIS-text-large/{LLM_model}/{middle_part}/
                llm_model_clean = row['llm_model'].replace(':', '_')
                save_dir = os.path.join(save_base_path, 'CLIP_evaluation', 'TRELLIS-text-large', llm_model_clean, middle_part)
            
            # Render from multiple viewpoints using Blender
            rendered_images = self.render_asset_multiview(asset_path, save_dir)
            if not rendered_images:
                result['error'] = 'Rendering failed'
                return result
            
            result['num_views_rendered'] = len(rendered_images)
            
            # Extract features
            image_features = self.extract_image_features(rendered_images)
            text_features = self.extract_text_features([text_prompt])
            
            # Calculate CLIP score
            clip_score = self.calculate_clip_score(image_features, text_features)
            
            result['clip_score'] = clip_score
            result['success'] = True
            
            # Save CLIP score to txt file in the same directory as rendered images
            if save_dir:
                clip_score_path = os.path.join(save_dir, 'clip_score.txt')
                with open(clip_score_path, 'w') as f:
                    f.write(f"{clip_score:.6f}\n")
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def evaluate_trellis_generated_dataset(self, 
                                         results_excel_path: str, 
                                         output_base_path: str, 
                                         save_base_path: str = None,
                                         max_assets: int = None,
                                         llm_models_filter: List[str] = None) -> Dict:
        """
        Evaluate CLIP scores for TRELLIS-generated 3D assets.
        
        Args:
            results_excel_path: Path to Excel file with LLM results
            output_base_path: Base path to generated outputs directory
            save_base_path: Base directory path to save results and rendered images
            max_assets: Maximum number of assets to evaluate (for testing)
            llm_models_filter: List of LLM models to evaluate (None for all)
            
        Returns:
            Dictionary with aggregated results
        """
        # Load LLM results
        results_df = self.load_llm_results(results_excel_path)
        
        # Filter by LLM models if specified
        if llm_models_filter:
            results_df = results_df[results_df['category'].isin(llm_models_filter)]
            print(f"Filtered to {len(results_df)} entries for models: {llm_models_filter}")
        
        if max_assets:
            results_df = results_df.head(max_assets)
            print(f"Limiting evaluation to first {max_assets} assets")
        
        # Evaluate each asset
        evaluation_results = []
        successful_evaluations = 0
        total_clip_score = 0.0
        
        # Track results by LLM model
        model_results = {}
        
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Evaluating generated assets"):
            result = self.evaluate_single_generated_asset(output_base_path, row, save_base_path)
            evaluation_results.append(result)
            
            llm_model = result['llm_model']
            if llm_model not in model_results:
                model_results[llm_model] = {'total': 0, 'successful': 0, 'total_score': 0.0}
            
            model_results[llm_model]['total'] += 1
            
            if result['success']:
                successful_evaluations += 1
                total_clip_score += result['clip_score']
                model_results[llm_model]['successful'] += 1
                model_results[llm_model]['total_score'] += result['clip_score']
            
            # Print progress every 10 assets
            if (idx + 1) % 10 == 0:
                current_success_rate = successful_evaluations / (idx + 1)
                current_avg_score = total_clip_score / successful_evaluations if successful_evaluations > 0 else 0
                print(f"Progress: {idx + 1}/{len(results_df)}, "
                      f"Success rate: {current_success_rate:.2%}, "
                      f"Avg CLIP score: {current_avg_score:.4f}")
        
        # Calculate aggregated metrics
        mean_clip_score = total_clip_score / successful_evaluations if successful_evaluations > 0 else 0.0
        mean_clip_score_scaled = mean_clip_score * 100  # Scale by 100 as mentioned in instructions
        
        # Calculate per-model metrics
        model_summaries = {}
        for model, stats in model_results.items():
            if stats['successful'] > 0:
                model_avg_score = stats['total_score'] / stats['successful']
                model_summaries[model] = {
                    'total_assets': stats['total'],
                    'successful_evaluations': stats['successful'],
                    'success_rate': stats['successful'] / stats['total'],
                    'mean_clip_score': model_avg_score,
                    'mean_clip_score_scaled': model_avg_score * 100
                }
            else:
                model_summaries[model] = {
                    'total_assets': stats['total'],
                    'successful_evaluations': 0,
                    'success_rate': 0.0,
                    'mean_clip_score': 0.0,
                    'mean_clip_score_scaled': 0.0
                }
        
        aggregated_results = {
            'mean_clip_score': mean_clip_score,
            'mean_clip_score_scaled': mean_clip_score_scaled,
            'total_assets': len(results_df),
            'successful_evaluations': successful_evaluations,
            'success_rate': successful_evaluations / len(results_df) if len(results_df) > 0 else 0.0,
            'results_excel_path': results_excel_path,
            'output_base_path': output_base_path,
            'model_summaries': model_summaries
        }
        
        print(f"\n=== TRELLIS Generated Assets CLIP Score Evaluation Results ===")
        print(f"Results file: {results_excel_path}")
        print(f"Output path: {output_base_path}")
        print(f"Total assets: {len(results_df)}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Success rate: {aggregated_results['success_rate']:.2%}")
        print(f"Mean CLIP Score: {mean_clip_score:.4f}")
        print(f"Mean CLIP Score (×100): {mean_clip_score_scaled:.2f}")
        
        print(f"\n=== Per-Model Results ===")
        for model, summary in model_summaries.items():
            print(f"{model}: {summary['successful_evaluations']}/{summary['total_assets']} "
                  f"({summary['success_rate']:.1%}) - "
                  f"CLIP Score: {summary['mean_clip_score']:.4f} "
                  f"(×100: {summary['mean_clip_score_scaled']:.2f})")
        
        # Save detailed results to CSV
        if save_base_path:
            # Generate CSV filename from Excel filename
            excel_filename = os.path.basename(results_excel_path)
            excel_name = os.path.splitext(excel_filename)[0]  # Remove .xlsx extension
            csv_filename = f"CLIP_evaluation_{excel_name}.csv"
            save_path = os.path.join(save_base_path, csv_filename)
            
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
        description='CLIP Score Evaluation for TRELLIS Generated 3D Assets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 5 assets using default paths
  python trellis_generated_clip_evaluator.py --max_assets 5
  
  # Use different Excel file and output directory
  python trellis_generated_clip_evaluator.py \\
    --results_excel /path/to/your/results.xlsx \\
    --output_base_path /path/to/your/generated/assets \\
    --save_path my_results.csv
  
  # Evaluate specific LLM models only
  python trellis_generated_clip_evaluator.py \\
    --llm_models gemma3 qwen3 \\
    --save_path model_comparison.csv
  
  # Use different Excel parts
  python trellis_generated_clip_evaluator.py \\
    --results_excel /mnt/nas/tmp/nayeon/sampled_data_100_random_results_part02.xlsx \\
    --save_path part02_results.csv
        """
    )
    
    # Required/Optional paths
    parser.add_argument('--results_excel', 
                        type=str, 
                        default='/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx',
                        help='Path to Excel file with LLM augmented prompts (default: part01.xlsx)')
    
    parser.add_argument('--output_base_path', 
                        type=str, 
                        default='/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816',
                        help='Base path to directory containing generated 3D assets')
    
    parser.add_argument('--save_path', 
                        type=str, 
                        default=None,
                        help='Base directory path to save evaluation results and rendered images')
    
    # Model and evaluation parameters
    parser.add_argument('--clip_model', 
                        type=str, 
                        default='openai/clip-vit-base-patch32',
                        help='CLIP model name for feature extraction')
    
    parser.add_argument('--max_assets', 
                        type=int, 
                        default=None,
                        help='Maximum number of assets to evaluate (useful for testing)')
    
    parser.add_argument('--llm_models', 
                        type=str, 
                        nargs='*', 
                        default=None,
                        help='Specific LLM model categories to evaluate (e.g., gemma3 qwen3 deepseek)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.results_excel):
        print(f"❌ Error: Excel file not found: {args.results_excel}")
        print("Available Excel files:")
        excel_dir = os.path.dirname(args.results_excel)
        if os.path.exists(excel_dir):
            for f in os.listdir(excel_dir):
                if f.endswith('.xlsx') and 'results' in f:
                    print(f"  - {os.path.join(excel_dir, f)}")
        return
    
    if not os.path.exists(args.output_base_path):
        print(f"❌ Error: Output base path not found: {args.output_base_path}")
        print("Please check the path to your generated 3D assets directory.")
        return
    
    print(f"📊 TRELLIS Generated Assets CLIP Score Evaluation")
    print(f"📁 Excel file: {args.results_excel}")
    print(f"📁 Assets path: {args.output_base_path}")
    if args.save_path:
        print(f"💾 Results will be saved to: {args.save_path}")
    else:
        print(f"💾 Results will not be saved (no save_path provided)")
    if args.llm_models:
        print(f"🔍 Evaluating models: {', '.join(args.llm_models)}")
    if args.max_assets:
        print(f"⚡ Limited to {args.max_assets} assets for testing")
    print()
    
    # Initialize evaluator
    evaluator = TrellisGeneratedCLIPEvaluator(clip_model_name=args.clip_model)
    
    # Run evaluation
    results = evaluator.evaluate_trellis_generated_dataset(
        args.results_excel, args.output_base_path, args.save_path, 
        args.max_assets, args.llm_models
    )
    
    print(f"\n🎯 Final CLIP Score (×100): {results['mean_clip_score_scaled']:.2f}")
    if args.save_path:
        excel_name = os.path.splitext(os.path.basename(args.results_excel))[0]
        csv_filename = f"CLIP_evaluation_{excel_name}.csv"
        final_save_path = os.path.join(args.save_path, csv_filename)
        print(f"✅ Results saved to: {final_save_path}")


if __name__ == "__main__":
    main()