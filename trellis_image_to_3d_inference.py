"""
=== AI/ML Solution Design (Image-to-3D) ===

📋 Problem Analysis:
- Domain: Image-to-3D Generation Model Testing Automation
- Task Type: Batch Inference and Result Management
- Data Characteristics: CSV + Image files input, Multiple 3D output formats
- Constraints: Image file management, Detailed timing measurements

🎯 Solution Design:
- Algorithm: Enhanced TRELLIS Image-to-3D pipeline with detailed timing
- Architecture: CSV + matched_image folder → Batch processing → CSV result tracking
- Data Pipeline: pandas CSV reading + PIL image loading, multi-stage timing measurement
- Evaluation Metrics: Generation time breakdown, success rate tracking

⚙️ Implementation Plan:
- Development Stages: Image processing → Timing enhancement → CSV output
- Technology Stack: pandas, PIL, opencv-python, time measurement
- Infrastructure: Existing environment maintained + matched_image folder
- Monitoring: CSV-based comprehensive result tracking
"""

import os
import pandas as pd
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import torch
from PIL import Image

os.environ['ATTN_BACKEND'] = 'xformers'

import imageio
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


class TrellisImageInferenceManager:
    """TRELLIS Image-to-3D model-based generation and result management class"""
    
    def __init__(self, model_name: str = "TRELLIS-image-large"):
        """
        Args:
            model_name: TRELLIS model name to use
        """
        self.model_name = model_name
        self.pipeline = None
        self.results_data: List[Dict] = []
        self.object_name_counter: Dict[str, int] = {}
        
        # Setup output folder
        today = datetime.now().strftime("%Y%m%d")
        self.output_base_dir = Path(f"output/{model_name}/{today}")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logs folder
        self.logs_dir = Path("output/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup matched_image folder
        self.matched_image_dir = Path("input_data/matched_image")
        self.matched_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_filename = None
        self.setup_logging()
        
        # Create default.yaml if it doesn't exist
        self.create_default_yaml()
        
        print(f"Results will be saved to: {self.output_base_dir}")
        print(f"Place input images in: {self.matched_image_dir}")
    
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d_%H%M%S")
        self.log_filename = self.logs_dir / f"{time_str}_pending_objs.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"=== TRELLIS Image-to-3D Inference Session Started ===")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Output directory: {self.output_base_dir}")
        logging.info(f"Image directory: {self.matched_image_dir}")
    
    def update_log_filename(self, object_count: int) -> None:
        """Update log filename with actual object count"""
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d_%H%M%S")
        new_log_filename = self.logs_dir / f"{time_str}_{object_count}objs_img2d.log"
        
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
        
        if self.log_filename and self.log_filename.exists():
            self.log_filename.rename(new_log_filename)
        
        file_handler = logging.FileHandler(new_log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        self.log_filename = new_log_filename
        logging.info(f"Log filename updated: {new_log_filename}")
    
    def create_default_yaml(self) -> None:
        """Create default.yaml configuration file for image-to-3D"""
        default_yaml_path = Path("default_image.yaml")
        if not default_yaml_path.exists():
            default_config = {
                'sparse_structure_sampler_params': {
                    'steps': 12,
                    'cfg_strength': 7.5
                },
                'slat_sampler_params': {
                    'steps': 12,
                    'cfg_strength': 3.0
                },
                'postprocessing': {
                    'simplify': 0.95,
                    'texture_size': 1024
                }
            }
            
            import yaml
            with open(default_yaml_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            logging.info(f"📄 Created default_image.yaml configuration file")
    
    def load_yaml_config(self, yaml_filename: str) -> Optional[Dict]:
        """Load YAML configuration file"""
        try:
            import yaml
            yaml_path = Path(yaml_filename)
            
            if not yaml_path.exists():
                logging.error(f"❌ YAML file not found: {yaml_filename}")
                return None
            
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logging.info(f"📄 Loaded YAML config: {yaml_filename}")
            return config
            
        except Exception as e:
            logging.error(f"❌ Failed to load YAML config {yaml_filename}: {e}")
            return None
    
    def load_pipeline(self) -> None:
        """Load TRELLIS Image-to-3D pipeline with error handling"""
        logging.info("Loading TRELLIS Image-to-3D pipeline...")
        try:
            self.pipeline = TrellisImageTo3DPipeline.from_pretrained(f"microsoft/{self.model_name}")
            self.pipeline.cuda()
            logging.info("Image-to-3D Pipeline loaded successfully!")
        except Exception as e:
            logging.error(f"Failed to load pipeline: {e}")
            raise
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Load object information from CSV file with image path validation"""
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['object_name', 'matched_image']
            
            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove empty values from required columns
            df = df.dropna(subset=required_columns)
            
            # Validate image files exist
            valid_rows = []
            for idx, row in df.iterrows():
                image_path = self.matched_image_dir / row['matched_image']
                if image_path.exists():
                    valid_rows.append(row)
                else:
                    logging.warning(f"⚠️  Image file not found: {image_path}, skipping {row['object_name']}")
            
            if not valid_rows:
                raise ValueError("No valid image files found")
            
            df = pd.DataFrame(valid_rows).reset_index(drop=True)
            
            # Add optional columns
            if 'seed' not in df.columns:
                df['seed'] = None
                logging.info("ℹ️  No 'seed' column found, will generate random seeds")
            
            if 'params' not in df.columns:
                df['params'] = None
                logging.info("ℹ️  No 'params' column found, will use default_image.yaml")
            
            logging.info(f"CSV file loaded successfully: {len(df)} objects with valid images")
            
            # Update log filename with actual object count
            self.update_log_filename(len(df))
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to load CSV file: {e}")
            raise
    
    def safe_filename(self, name: str, max_length: int = 50) -> str:
        """Generate safe filename with automatic conversion"""
        import re
        
        safe_name = str(name).lower().strip()
        safe_name = re.sub(r'[^\w\-_.]', '_', safe_name)
        safe_name = re.sub(r'_+', '_', safe_name)
        safe_name = safe_name.strip('_')
        safe_name = safe_name[:max_length]
        
        if not safe_name:
            safe_name = "unnamed_object"
        
        return safe_name
    
    def load_and_validate_image(self, image_path: Path) -> Optional[Image.Image]:
        """Load and validate image file"""
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logging.info(f"📸 Loaded image: {image_path.name} ({image.size})")
            return image
            
        except Exception as e:
            logging.error(f"❌ Failed to load image {image_path}: {e}")
            return None
    
    def extract_thumbnail(self, video_path: Path, thumbnail_path: Path) -> bool:
        """Save middle frame of video as thumbnail"""
        try:
            reader = imageio.get_reader(str(video_path))
            
            try:
                frame_count = reader.get_length()
                if frame_count == float('inf') or frame_count <= 0:
                    frame_count = 300
                frame_count = int(frame_count)
            except (ValueError, OverflowError, TypeError):
                frame_count = 300
            
            middle_frame_idx = min(frame_count // 2, frame_count - 1)
            middle_frame_idx = max(0, middle_frame_idx)
            
            middle_frame = reader.get_data(middle_frame_idx)
            reader.close()
            
            imageio.imwrite(str(thumbnail_path), middle_frame)
            return True
            
        except Exception as e:
            logging.error(f"Failed to create thumbnail ({video_path}): {e}")
            return False
    
    def run_pipeline_with_timing(self, image: Image.Image, seed: int, config: Dict) -> tuple:
        """Run image-to-3D pipeline with detailed timing measurement"""
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        timing_info = {}
        total_start = time.time()
        
        # Extract parameters from config
        sparse_params = config.get('sparse_structure_sampler_params', {})
        slat_params = config.get('slat_sampler_params', {})
        
        try:
            with torch.no_grad():
                preprocessed_image = self.pipeline.preprocess_image(image)

                # Step 1: Condition encoding (image processing)
                cond_start = time.time()
                cond = self.pipeline.get_cond([preprocessed_image])
                timing_info['condition_encoding_time'] = time.time() - cond_start
                
                # Set seed
                torch.manual_seed(seed)
                
                # Step 2: Sparse structure sampling
                sparse_start = time.time()
                coords = self.pipeline.sample_sparse_structure(cond, 1, sparse_params)
                timing_info['sparse_structure_time'] = time.time() - sparse_start
                
                # Step 3: SLAT sampling
                slat_start = time.time()
                slat = self.pipeline.sample_slat(cond, coords, slat_params)
                timing_info['slat_sampling_time'] = time.time() - slat_start
                
                # Step 4: Decoding
                decode_start = time.time()
                outputs = self.pipeline.decode_slat(slat, ['mesh', 'gaussian', 'radiance_field'])
                timing_info['decoding_time'] = time.time() - decode_start
                
                timing_info['total_pipeline_time'] = time.time() - total_start
            
            return outputs, timing_info, preprocessed_image
            
        except AttributeError as e:
            logging.warning(f"⚠️  Individual timing not available, using run() method: {e}")
            
            outputs = self.pipeline.run(
                image,
                seed=seed,
                num_samples=1,
                formats=['mesh', 'gaussian', 'radiance_field'],
                sparse_structure_sampler_params=sparse_params,
                slat_sampler_params=slat_params
            )
            
            timing_info['total_pipeline_time'] = time.time() - total_start
            timing_info['condition_encoding_time'] = 0
            timing_info['sparse_structure_time'] = 0
            timing_info['slat_sampling_time'] = 0
            timing_info['decoding_time'] = 0

            preprocessed_image = self.pipeline.preprocess_image(image)
            
            return outputs, timing_info, preprocessed_image
    
    def generate_single_object(self, object_name: str, image_path: str, seed: int, config: Dict, params_filename: str) -> Dict:
        """Generate single object from image"""
        start_time = time.time()
        generation_time = datetime.now()
        
        logging.info(f"Generating: {object_name} - (Image: {image_path})")
        
        try:
            # Load and validate image
            full_image_path = self.matched_image_dir / image_path
            image = self.load_and_validate_image(full_image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # 3D object generation with detailed timing and custom config
            outputs, timing_info, preprocessed_image = self.run_pipeline_with_timing(image, seed, config)
            
            # Video rendering time
            render_start = time.time()
            video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
            video_rf = render_utils.render_video(outputs['radiance_field'][0])['color']
            video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
            render_time = time.time() - render_start
            
            # Generate filename
            safe_name = self.safe_filename(object_name)
            
            # Create object-specific folder with seed subfolder
            object_base_dir = self.output_base_dir / f"{safe_name}"
            seed_dir = object_base_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            # Save preprocessed images
            preprocessed_filename = f"{safe_name}_preprocessed.png"
            preprocessed_image.save(seed_dir / preprocessed_filename, "PNG")
            
            # File saving time
            save_start = time.time()
            
            # Video file names
            video_files = {
                'gs': f"{safe_name}_gs.mp4",
                'rf': f"{safe_name}_rf.mp4", 
                'mesh': f"{safe_name}_mesh.mp4"
            }
            
            # Save video files
            imageio.mimsave(seed_dir / video_files['gs'], video_gs, fps=30)
            imageio.mimsave(seed_dir / video_files['rf'], video_rf, fps=30)
            imageio.mimsave(seed_dir / video_files['mesh'], video_mesh, fps=30)
            
            # Save GLB file with custom config
            glb_filename = f"{safe_name}.glb"
            postprocessing_config = config.get('postprocessing', {})
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify=postprocessing_config.get('simplify', 0.95),
                texture_size=postprocessing_config.get('texture_size', 1024),
            )
            glb.export(seed_dir / glb_filename)
            
            # Save PLY file
            ply_filename = f"{safe_name}.ply"
            outputs['gaussian'][0].save_ply(seed_dir / ply_filename)
            
            # Generate thumbnail
            thumbnail_filename = f"{safe_name}_{seed}.jpg"
            thumbnail_success = self.extract_thumbnail(
                seed_dir / video_files['gs'],
                object_base_dir / thumbnail_filename
            )
            
            save_time = time.time() - save_start
            total_time = time.time() - start_time
            
            # Construct result information
            result_info = {
                'object_name': object_name,
                'seed': seed,
                'params': params_filename,
                'matched_image': image_path,  # 추가된 컬럼
                'total_generation_time_sec': round(total_time, 2),
                'pipeline_execution_time_sec': round(timing_info['total_pipeline_time'], 2),
                'condition_encoding_time_sec': round(timing_info['condition_encoding_time'], 2),
                'sparse_structure_time_sec': round(timing_info['sparse_structure_time'], 2),
                'slat_sampling_time_sec': round(timing_info['slat_sampling_time'], 2),
                'decoding_time_sec': round(timing_info['decoding_time'], 2),
                'rendering_time_sec': round(render_time, 2),
                'file_saving_time_sec': round(save_time, 2),
                'generation_date_kst': generation_time.strftime('%Y-%m-%d %H:%M:%S'),
                'save_path': str(seed_dir),
                'glb_filename': glb_filename,
                'ply_filename': ply_filename,
                'gs_video_filename': video_files['gs'],
                'rf_video_filename': video_files['rf'],
                'mesh_video_filename': video_files['mesh'],
                'thumbnail_filename': thumbnail_filename if thumbnail_success else '',
                'generation_status': 'success'
            }
            
            logging.info(f"✅ Generation completed: {object_name} ({total_time:.1f}s)")
            return result_info
            
        except Exception as e:
            error_time = time.time() - start_time
            logging.error(f"❌ Generation failed: {object_name} - {e}")
            
            return {
                'object_name': object_name,
                'seed': seed,
                'params': params_filename,
                'matched_image': image_path,
                'total_generation_time_sec': round(error_time, 2),
                'pipeline_execution_time_sec': 0,
                'condition_encoding_time_sec': 0,
                'sparse_structure_time_sec': 0,
                'slat_sampling_time_sec': 0,
                'decoding_time_sec': 0,
                'rendering_time_sec': 0,
                'file_saving_time_sec': 0,
                'generation_date_kst': generation_time.strftime('%Y-%m-%d %H:%M:%S'),
                'save_path': '',
                'glb_filename': '',
                'ply_filename': '',
                'gs_video_filename': '',
                'rf_video_filename': '',
                'mesh_video_filename': '',
                'thumbnail_filename': '',
                'generation_status': f'failed: {str(e)}'
            }
    
    def process_batch(self, csv_path: str, start_seed: int = 1) -> None:
        """Execute batch processing for image-to-3D generation"""
        import random
        
        if self.pipeline is None:
            self.load_pipeline()
        print(csv_path)
        df = self.load_csv_data(csv_path)
        
        logging.info(f"\n🚀 Starting Image-to-3D batch processing: {len(df)} objects")
        logging.info("=" * 60)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            object_name = str(row['object_name'])
            image_path = str(row['matched_image'])
            
            # Determine seed value
            has_seed_column = 'seed' in df.columns
            if has_seed_column:
                seed_cell_value = row['seed']
                is_valid_seed = bool(pd.notna(seed_cell_value))
                
                if is_valid_seed:
                    seed_value = str(seed_cell_value).strip()
                    if seed_value:
                        try:
                            seed = int(float(seed_value))
                            logging.info(f"🎲 Using user-provided seed: {seed}")
                        except (ValueError, TypeError):
                            seed = random.randint(0, 2147483647)
                            logging.warning(f"⚠️  Invalid seed value '{seed_cell_value}', using random seed: {seed}")
                    else:
                        seed = random.randint(0, 2147483647)
                        logging.info(f"🎲 Generated random seed: {seed}")
                else:
                    seed = random.randint(0, 2147483647)
                    logging.info(f"🎲 Generated random seed: {seed}")
            else:
                seed = random.randint(0, 2147483647)
                logging.info(f"🎲 Generated random seed: {seed}")
            
            # Determine YAML config file
            has_params_column = 'params' in df.columns
            if has_params_column:
                params_cell_value = row['params']
                is_valid_params = bool(pd.notna(params_cell_value))
                
                if is_valid_params:
                    params_value = str(params_cell_value).strip()
                    if params_value:
                        params_filename = params_value
                        config = self.load_yaml_config(params_filename)
                        if config is None:
                            logging.warning(f"⚠️  Failed to load {params_filename}, skipping {object_name}")
                            continue
                        logging.info(f"📄 Using custom YAML: {params_filename}")
                    else:
                        params_filename = "default_image.yaml"
                        config = self.load_yaml_config(params_filename)
                        logging.info(f"📄 Using default YAML: {params_filename}")
                else:
                    params_filename = "default_image.yaml"
                    config = self.load_yaml_config(params_filename)
                    logging.info(f"📄 Using default YAML: {params_filename}")
            else:
                params_filename = "default_image.yaml"
                config = self.load_yaml_config(params_filename)
                logging.info(f"📄 Using default YAML: {params_filename}")
            
            if config is None:
                logging.warning(f"⚠️  Failed to load config, skipping {object_name}")
                continue
            
            result = self.generate_single_object(object_name, image_path, seed, config, params_filename)
            self.results_data.append(result)
            
            logging.info(f"Progress: {i + 1}/{len(df)}")
            logging.info("-" * 40)
        
        self.save_results_csv()
        
        logging.info("\n🎉 All Image-to-3D objects generated successfully!")
        self.print_summary()
    
    def save_results_csv(self) -> None:
        """Save results to CSV file with accumulation and sorting"""
        if not self.results_data:
            logging.info("No results to save.")
            return
        
        csv_path = self.output_base_dir / "generation_results.csv"
        
        existing_data = []
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                existing_data = existing_df.to_dict('records')
                logging.info(f"📄 Loaded existing results: {len(existing_data)} records")
            except Exception as e:
                logging.warning(f"Warning: Could not load existing CSV: {e}")
        
        all_data = existing_data + self.results_data
        results_df = pd.DataFrame(all_data)
        results_df = results_df.sort_values(['object_name', 'seed'], ascending=[True, True])
        results_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logging.info(f"📊 Results CSV saved: {csv_path} ({len(all_data)} total records)")
        logging.info(f"   - Previous records: {len(existing_data)}")
        logging.info(f"   - New records: {len(self.results_data)}")
    
    def print_summary(self) -> None:
        """Print generation results summary"""
        if not self.results_data:
            return
        
        df = pd.DataFrame(self.results_data)
        successful = df[df['generation_status'] == 'success']
        failed = df[df['generation_status'] != 'success']
        
        logging.info("\n📈 Image-to-3D Generation Results Summary:")
        logging.info(f"✅ Successful: {len(successful)} objects")
        logging.info(f"❌ Failed: {len(failed)} objects")
        
        if len(successful) > 0:
            avg_total_time = successful['total_generation_time_sec'].mean()
            avg_pipeline_time = successful['pipeline_execution_time_sec'].mean()
            total_time = successful['total_generation_time_sec'].sum()
            logging.info(f"⏱️  Average total time: {avg_total_time:.1f}s")
            logging.info(f"⏱️  Average pipeline time: {avg_pipeline_time:.1f}s")
            logging.info(f"⏱️  Total generation time: {total_time:.1f}s")
            
            logging.info("\n📊 Timing Breakdown (Average):")
            logging.info(f"  - Condition encoding: {successful['condition_encoding_time_sec'].mean():.2f}s")
            logging.info(f"  - Sparse structure: {successful['sparse_structure_time_sec'].mean():.2f}s")
            logging.info(f"  - SLAT sampling: {successful['slat_sampling_time_sec'].mean():.2f}s")
            logging.info(f"  - Decoding: {successful['decoding_time_sec'].mean():.2f}s")
            logging.info(f"  - Rendering: {successful['rendering_time_sec'].mean():.2f}s")
            logging.info(f"  - File saving: {successful['file_saving_time_sec'].mean():.2f}s")
        
        if len(failed) > 0:
            logging.info(f"\n❌ Failed objects:")
            for _, row in failed.iterrows():
                logging.info(f"  - {row['object_name']}: {row['generation_status']}")
        
        logging.info(f"\n📁 Results saved to: {self.output_base_dir}")


def main():
    """Main execution function for Image-to-3D generation"""
    csv_file = "input_data/input_objects.csv"  # Input CSV file path
    
    # Create manager
    manager = TrellisImageInferenceManager()
    
    # Execute batch processing
    try:
        manager.process_batch(csv_file, start_seed=1)
    except FileNotFoundError:
        logging.error(f"❌ CSV file not found: {csv_file}")
        logging.info("📝 Please prepare a CSV file with the following format:")
        logging.info("   - object_name: Object name (English only)")
        logging.info("   - matched_image: Image filename in input_data/matched_image/ folder")
        logging.info("   - seed: Random seed (optional, leave empty for random generation)")
        logging.info("   - params: YAML config file (optional, leave empty for default_image.yaml)")
        
        # Create sample CSV file
        sample_data = {
            'object_name': [
                'cat_statue',
                'wooden_chair',
                'vintage_clock',
                'ceramic_vase',
                'metal_lamp'
            ],
            'matched_image': [
                'cat_statue.jpg',
                'wooden_chair.png',
                'vintage_clock.jpg',
                'ceramic_vase.png',
                'metal_lamp.jpg'
            ],
            'seed': [
                42,
                '',  # Empty - will use random
                1234567890,
                '',  # Empty - will use random  
                999
            ],
            'params': [
                'default_image.yaml',
                '',  # Empty - will use default_image.yaml
                'high_quality_image.yaml',
                'default_image.yaml',
                ''  # Empty - will use default_image.yaml
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        
        # Create input_data directory if it doesn't exist
        Path("input_data").mkdir(exist_ok=True)
        sample_df.to_csv('input_data/sample_input_image.csv', index=False)
        logging.info("📋 Sample file created: input_data/sample_input_image.csv")
        logging.info("📂 Don't forget to place image files in input_data/matched_image/ folder!")


if __name__ == "__main__":
    main()