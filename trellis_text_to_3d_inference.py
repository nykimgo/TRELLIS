"""
=== AI/ML Solution Design ===

📋 Problem Analysis:
- Domain: 3D Generation Model Testing Automation
- Task Type: Batch Inference and Result Management
- Data Characteristics: CSV input, Multiple 3D output formats
- Constraints: English-only naming, Detailed timing measurements

🎯 Solution Design:
- Algorithm: Enhanced TRELLIS pipeline with detailed timing
- Architecture: CSV input → Batch processing → CSV result tracking
- Data Pipeline: pandas CSV reading, multi-stage timing measurement
- Evaluation Metrics: Generation time breakdown, success rate tracking

⚙️ Implementation Plan:
- Development Stages: Code refactoring → Timing enhancement → CSV output
- Technology Stack: pandas, opencv-python, time measurement
- Infrastructure: Existing environment maintained
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

os.environ['ATTN_BACKEND'] = 'xformers'

import imageio
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


class TrellisInferenceManager:
    """TRELLIS model-based 3D generation and result management class"""
    
    def __init__(self, model_name: str = "TRELLIS-text-xlarge"):
        """
        Args:
            model_name: TRELLIS model name to use
        """
        self.model_name = model_name
        self.pipeline = None
        self.results_data: List[Dict] = []
        self.object_name_counter: Dict[str, int] = {}  # Fix: Added missing initialization
        
        # Setup output folder
        today = datetime.now().strftime("%Y%m%d")
        self.output_base_dir = Path(f"output/{model_name}/{today}")
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logs folder
        self.logs_dir = Path("output/logs")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.log_filename = None
        self.setup_logging()
        
        # Create default.yaml if it doesn't exist
        self.create_default_yaml()
        
        print(f"Results will be saved to: {self.output_base_dir}")
    
    def setup_logging(self) -> None:
        """Setup logging configuration"""
        # Generate log filename when logging is initialized
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d_%H%M%S")
        self.log_filename = self.logs_dir / f"{time_str}_pending_objs.log"
        
        # Setup logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename, encoding='utf-8'),
                logging.StreamHandler()  # Also output to console
            ]
        )
        
        logging.info(f"=== TRELLIS Inference Session Started ===")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Output directory: {self.output_base_dir}")
    
    def update_log_filename(self, object_count: int) -> None:
        """Update log filename with actual object count"""
        current_time = datetime.now()
        time_str = current_time.strftime("%Y%m%d_%H%M%S")
        new_log_filename = self.logs_dir / f"{time_str}_{object_count}objs.log"
        
        # Get current file handler
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
        
        # Rename the file and add new handler
        if self.log_filename and self.log_filename.exists():
            self.log_filename.rename(new_log_filename)
        
        # Add new file handler
        file_handler = logging.FileHandler(new_log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        self.log_filename = new_log_filename
        logging.info(f"Log filename updated: {new_log_filename}")
    
    def create_default_yaml(self) -> None:
        """Create default.yaml configuration file if it doesn't exist"""
        default_yaml_path = Path("default.yaml")
        if not default_yaml_path.exists():
            default_config = {
                'sparse_structure_sampler_params': {
                    'steps': 25,
                    'cfg_strength': 7.5
                },
                'slat_sampler_params': {
                    'steps': 25,
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
            logging.info(f"📄 Created default.yaml configuration file")
    
    def load_yaml_config(self, yaml_filename: str) -> Optional[Dict]:
        """
        Load YAML configuration file
        
        Args:
            yaml_filename: YAML file name
            
        Returns:
            Configuration dictionary or None if failed
        """
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
        """Load TRELLIS pipeline with error handling"""
        logging.info("Loading TRELLIS pipeline...")
        try:
            self.pipeline = TrellisTextTo3DPipeline.from_pretrained(f"microsoft/{self.model_name}")
            self.pipeline.cuda()
            logging.info("Pipeline loaded successfully!")
        except Exception as e:
            logging.error(f"Failed to load pipeline: {e}")
            raise
    
    def load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load object information from CSV file
        
        Args:
            csv_path: CSV file path
            
        Returns:
            DataFrame containing object information
        """
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['object_name', 'text_prompt']
            
            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove empty values from required columns
            df = df.dropna(subset=required_columns)
            
            # Add seed column if not exists
            if 'seed' not in df.columns:
                df['seed'] = None
                logging.info("ℹ️  No 'seed' column found, will generate random seeds")
            
            # Add params column if not exists
            if 'params' not in df.columns:
                df['params'] = None
                logging.info("ℹ️  No 'params' column found, will use default.yaml")
            
            logging.info(f"CSV file loaded successfully: {len(df)} objects")
            
            # Update log filename with actual object count
            self.update_log_filename(len(df))
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to load CSV file: {e}")
            raise

    def get_unique_object_name(self, base_name: str) -> str:
        """
        Generate unique object name to handle duplicates
        
        Args:
            base_name: Base object name
            
        Returns:
            Unique object name with counter if needed
        """
        # Count occurrences
        if base_name in self.object_name_counter:
            self.object_name_counter[base_name] += 1
            unique_name = f"{base_name}_{self.object_name_counter[base_name]:02d}"
            print(f"🔄 Duplicate object name detected: '{base_name}' → '{unique_name}'")
        else:
            self.object_name_counter[base_name] = 0
            unique_name = base_name
        
        return unique_name

    def safe_filename(self, name: str, max_length: int = 50) -> str:
        """Generate safe filename with automatic conversion"""
        import re
        
        # 1. Convert to lowercase
        safe_name = str(name).lower().strip()
        
        # 2. Replace special characters and spaces with underscore
        safe_name = re.sub(r'[^\w\-_.]', '_', safe_name)
        
        # 3. Compress multiple underscores to single
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # 4. Remove leading and trailing underscores
        safe_name = safe_name.strip('_')
        
        # 5. Limit length
        safe_name = safe_name[:max_length]
        
        # 6. Use default if empty
        if not safe_name:
            safe_name = "unnamed_object"
        
        return safe_name
    
    def extract_thumbnail(self, video_path: Path, thumbnail_path: Path) -> bool:
        """
        Save middle frame of video as thumbnail
        
        Args:
            video_path: Video file path
            thumbnail_path: Thumbnail save path
            
        Returns:
            Success status
        """
        try:
            # Read video with imageio
            reader = imageio.get_reader(str(video_path))
        
            # Get total frame count safely
            try:
                frame_count = reader.get_length()
                # Handle special cases
                if frame_count == float('inf') or frame_count <= 0:
                    frame_count = 300  # Default for TRELLIS videos (300 frames)
                frame_count = int(frame_count)
            except (ValueError, OverflowError, TypeError):
                # If get_length() fails, use default
                frame_count = 300
            
            # Calculate middle frame index
            middle_frame_idx = min(frame_count // 2, frame_count - 1)
            middle_frame_idx = max(0, middle_frame_idx)  # Ensure non-negative
            
            # Get middle frame
            middle_frame = reader.get_data(middle_frame_idx)
            reader.close()
            
            # Save image
            imageio.imwrite(str(thumbnail_path), middle_frame)
            return True
            
        except Exception as e:
            logging.error(f"Failed to create thumbnail ({video_path}): {e}")
            return False
    
    def run_pipeline_with_timing(self, text_prompt: str, seed: int, config: Dict) -> tuple:
        """
        Run pipeline with detailed timing measurement and custom configuration
        
        Args:
            text_prompt: Text prompt for generation
            seed: Random seed
            config: YAML configuration dictionary
            
        Returns:
            (outputs, timing_info)
        """
        # Check if pipeline is loaded
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline() first.")
        
        timing_info = {}
        
        # Total pipeline time
        total_start = time.time()
        
        # Extract parameters from config
        sparse_params = config.get('sparse_structure_sampler_params', {})
        slat_params = config.get('slat_sampler_params', {})
        
        # Manual step-by-step execution for detailed timing
        try:
            with torch.no_grad():  # gradient 계산 비활성화
                # Step 1: Condition encoding
                cond_start = time.time()
                cond = self.pipeline.get_cond([text_prompt])
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
            
            return outputs, timing_info
            
        except AttributeError as e:
            # Fallback to run() method if individual methods are not accessible
            print(f"⚠️  Individual timing not available, using run() method: {e}")
            
            # Use the official run method with custom parameters
            outputs = self.pipeline.run(
                prompt=text_prompt,
                seed=seed,
                num_samples=1,
                formats=['mesh', 'gaussian', 'radiance_field'],
                sparse_structure_sampler_params=sparse_params,
                slat_sampler_params=slat_params
            )
            
            timing_info['total_pipeline_time'] = time.time() - total_start
            
            # Set other timing values to 0 since we can't measure individual steps
            timing_info['condition_encoding_time'] = 0
            timing_info['sparse_structure_time'] = 0
            timing_info['slat_sampling_time'] = 0
            timing_info['decoding_time'] = 0
            
            return outputs, timing_info
    
    def generate_single_object(self, object_name: str, text_prompt: str, seed: int, config: Dict, params_filename: str) -> Dict:
        """
        Generate single object
        
        Args:
            object_name: Object name
            text_prompt: Text prompt
            seed: Seed value
            config: YAML configuration dictionary
            params_filename: YAML filename used for generation
            
        Returns:
            Generation result information
        """
        start_time = time.time()
        generation_time = datetime.now()
        
        logging.info(f"Generating: {object_name} - {text_prompt}")
        
        try:
            # 3D object generation with detailed timing and custom config
            outputs, timing_info = self.run_pipeline_with_timing(text_prompt, seed, config)
            
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
                'total_generation_time_sec': round(total_time, 2),
                'pipeline_execution_time_sec': round(timing_info['total_pipeline_time'], 2),
                'condition_encoding_time_sec': round(timing_info['condition_encoding_time'], 2),
                'sparse_structure_time_sec': round(timing_info['sparse_structure_time'], 2),
                'slat_sampling_time_sec': round(timing_info['slat_sampling_time'], 2),
                'decoding_time_sec': round(timing_info['decoding_time'], 2),
                'rendering_time_sec': round(render_time, 2),
                'file_saving_time_sec': round(save_time, 2),
                'text_prompt': text_prompt,
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
                'total_generation_time_sec': round(error_time, 2),
                'pipeline_execution_time_sec': 0,
                'condition_encoding_time_sec': 0,
                'sparse_structure_time_sec': 0,
                'slat_sampling_time_sec': 0,
                'decoding_time_sec': 0,
                'rendering_time_sec': 0,
                'file_saving_time_sec': 0,
                'text_prompt': text_prompt,
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
        """
        Execute batch processing
        
        Args:
            csv_path: Input CSV file path
            start_seed: Starting seed value (used only when CSV doesn't have seed column or seed is empty)
        """
        import random
        
        # Load pipeline
        if self.pipeline is None:
            self.load_pipeline()
        
        # Load CSV data
        df = self.load_csv_data(csv_path)
        
        logging.info(f"\n🚀 Starting batch processing: {len(df)} objects")
        logging.info("=" * 60)
        
        # Process each object
        for i, (idx, row) in enumerate(df.iterrows()):
            object_name = str(row['object_name'])
            text_prompt = str(row['text_prompt'])
            
            # Determine seed value
            has_seed_column = 'seed' in df.columns
            if has_seed_column:
                seed_cell_value = row['seed']
                is_valid_seed = bool(pd.notna(seed_cell_value))
                
                if is_valid_seed:
                    seed_value = str(seed_cell_value).strip()
                    if seed_value:
                        # Use user-provided seed
                        try:
                            seed = int(float(seed_value))  # Handle both int and float inputs
                            print(f"🎲 Using user-provided seed: {seed}")
                        except (ValueError, TypeError):
                            # Invalid seed value, generate random
                            seed = random.randint(0, 2147483647)
                            print(f"⚠️  Invalid seed value '{seed_cell_value}', using random seed: {seed}")
                    else:
                        # Empty seed value, generate random
                        seed = random.randint(0, 2147483647)
                        print(f"🎲 Generated random seed: {seed}")
                else:
                    # NaN seed value, generate random
                    seed = random.randint(0, 2147483647)
                    print(f"🎲 Generated random seed: {seed}")
            else:
                # Generate random seed
                seed = random.randint(0, 2147483647)
                print(f"🎲 Generated random seed: {seed}")
            
            # Determine YAML config file
            has_params_column = 'params' in df.columns
            if has_params_column:
                params_cell_value = row['params']
                is_valid_params = bool(pd.notna(params_cell_value))
                
                if is_valid_params:
                    params_value = str(params_cell_value).strip()
                    if params_value:
                        # Use user-provided YAML file
                        params_filename = params_value
                        config = self.load_yaml_config(params_filename)
                        if config is None:
                            print(f"⚠️  Failed to load {params_filename}, skipping {object_name}")
                            # Add failed result
                            failed_result = {
                                'object_name': object_name,
                                'seed': seed,
                                'params': params_filename,
                                'total_generation_time_sec': 0,
                                'pipeline_execution_time_sec': 0,
                                'condition_encoding_time_sec': 0,
                                'sparse_structure_time_sec': 0,
                                'slat_sampling_time_sec': 0,
                                'decoding_time_sec': 0,
                                'rendering_time_sec': 0,
                                'file_saving_time_sec': 0,
                                'text_prompt': text_prompt,
                                'generation_date_kst': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'save_path': '',
                                'glb_filename': '',
                                'ply_filename': '',
                                'gs_video_filename': '',
                                'rf_video_filename': '',
                                'mesh_video_filename': '',
                                'thumbnail_filename': '',
                                'generation_status': f'failed: YAML file not found - {params_filename}'
                            }
                            self.results_data.append(failed_result)
                            print(f"Progress: {i + 1}/{len(df)}")
                            print("-" * 40)
                            continue
                        print(f"📄 Using custom YAML: {params_filename}")
                    else:
                        # Empty params value, use default
                        params_filename = "default.yaml"
                        config = self.load_yaml_config(params_filename)
                        print(f"📄 Using default YAML: {params_filename}")
                else:
                    # NaN params value, use default
                    params_filename = "default.yaml"
                    config = self.load_yaml_config(params_filename)
                    print(f"📄 Using default YAML: {params_filename}")
            else:
                # No params column, use default
                params_filename = "default.yaml"
                config = self.load_yaml_config(params_filename)
                print(f"📄 Using default YAML: {params_filename}")
            
            # If config loading failed, skip this object
            if config is None:
                print(f"⚠️  Failed to load config, skipping {object_name}")
                continue
            
            result = self.generate_single_object(object_name, text_prompt, seed, config, params_filename)
            self.results_data.append(result)
            
            print(f"Progress: {i + 1}/{len(df)}")
            print("-" * 40)
        
        # Save results CSV
        self.save_results_csv()
        
        print("\n🎉 All objects generated successfully!")
        self.print_summary()
    
    def save_results_csv(self) -> None:
        """Save results to CSV file with accumulation and sorting"""
        if not self.results_data:
            print("No results to save.")
            return
        
        csv_path = self.output_base_dir / "generation_results.csv"
        
        # Load existing data if file exists
        existing_data = []
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                existing_data = existing_df.to_dict('records')
                print(f"📄 Loaded existing results: {len(existing_data)} records")
            except Exception as e:
                print(f"Warning: Could not load existing CSV: {e}")
        
        # Combine existing and new data
        all_data = existing_data + self.results_data
        
        # Create DataFrame and sort
        results_df = pd.DataFrame(all_data)
        
        # Sort by object_name (alphabetical) then by seed (numerical)
        results_df = results_df.sort_values(['object_name', 'seed'], ascending=[True, True])
        
        # Save CSV file
        results_df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"📊 Results CSV saved: {csv_path} ({len(all_data)} total records)")
        print(f"   - Previous records: {len(existing_data)}")
        print(f"   - New records: {len(self.results_data)}")
    
    def print_summary(self) -> None:
        """Print generation results summary"""
        if not self.results_data:
            return
        
        df = pd.DataFrame(self.results_data)
        successful = df[df['generation_status'] == 'success']
        failed = df[df['generation_status'] != 'success']
        
        print("\n📈 Generation Results Summary:")
        print(f"✅ Successful: {len(successful)} objects")
        print(f"❌ Failed: {len(failed)} objects")
        
        if len(successful) > 0:
            avg_total_time = successful['total_generation_time_sec'].mean()
            avg_pipeline_time = successful['pipeline_execution_time_sec'].mean()
            total_time = successful['total_generation_time_sec'].sum()
            print(f"⏱️  Average total time: {avg_total_time:.1f}s")
            print(f"⏱️  Average pipeline time: {avg_pipeline_time:.1f}s")
            print(f"⏱️  Total generation time: {total_time:.1f}s")
            
            # Detailed timing breakdown
            print("\n📊 Timing Breakdown (Average):")
            print(f"  - Condition encoding: {successful['condition_encoding_time_sec'].mean():.2f}s")
            print(f"  - Sparse structure: {successful['sparse_structure_time_sec'].mean():.2f}s")
            print(f"  - SLAT sampling: {successful['slat_sampling_time_sec'].mean():.2f}s")
            print(f"  - Decoding: {successful['decoding_time_sec'].mean():.2f}s")
            print(f"  - Rendering: {successful['rendering_time_sec'].mean():.2f}s")
            print(f"  - File saving: {successful['file_saving_time_sec'].mean():.2f}s")
        
        if len(failed) > 0:
            print(f"\n❌ Failed objects:")
            for _, row in failed.iterrows():
                print(f"  - {row['object_name']}: {row['generation_status']}")
        
        print(f"\n📁 Results saved to: {self.output_base_dir}")


def main():
    """Main execution function"""
    # Usage example
    csv_file = "input_objects.csv"  # Input CSV file path
    
    # Create manager
    manager = TrellisInferenceManager()
    
    # Execute batch processing
    try:
        manager.process_batch(csv_file, start_seed=1)
    except FileNotFoundError:
        print(f"❌ CSV file not found: {csv_file}")
        print("📝 Please prepare a CSV file with the following format:")
        print("   - object_name: Object name (English only)")
        print("   - text_prompt: Text prompt for generation (English only)")
        print("   - seed: Random seed (optional, leave empty for random generation)")
        print("   - params: YAML config file (optional, leave empty for default.yaml)")
        
        # Create sample CSV file
        sample_data = {
            'object_name': [
                'vintage_camcorder',
                'lens_cap',
                'camera_lens',
                'human_hand',
                'retro_camera'
            ],
            'text_prompt': [
                'A vintage camcorder with buttons and lens',
                'A black circular lens cap for camera',
                'A camera lens with adjustable aperture', 
                'A human hand reaching forward',
                'A retro video camera from the 1990s'
            ],
            'seed': [
                42,
                '',  # Empty - will use random
                1234567890,
                '',  # Empty - will use random  
                999
            ],
            'params': [
                'default.yaml',
                '',  # Empty - will use default.yaml
                'custom_high_quality.yaml',
                'default.yaml',
                ''  # Empty - will use default.yaml
            ]
        }
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv('sample_input.csv', index=False)
        print("📋 Sample file created: sample_input.csv")


if __name__ == "__main__":
    main()