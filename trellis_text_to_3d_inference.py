import os
import pandas as pd
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import torch
import random

# TRELLIS 환경 설정 (임포트 전에 설정 필요)
os.environ['SPCONV_ALGO'] = 'native'
os.environ['ATTN_BACKEND'] = 'xformers'

try:
    import imageio
    from trellis.pipelines import TrellisTextTo3DPipeline
    from trellis.utils import render_utils, postprocessing_utils
    TRELLIS_AVAILABLE = True
except ImportError as e:
    print(f"❌ TRELLIS 모듈 임포트 실패: {e}")
    print("💡 TRELLIS 프로젝트 루트에서 실행하거나 PYTHONPATH를 설정하세요")
    TRELLIS_AVAILABLE = False


class TrellisInferenceManager:
    """TRELLIS model-based 3D generation and result management class"""
    
    def __init__(self, model_path: str = "microsoft/TRELLIS-text-xlarge", base_output_dir: str = "/mnt/nas/tmp/nayeon"):
        """
        Args:
            model_path: TRELLIS model path (local path or HuggingFace model name)
            base_output_dir: Base directory for output files
        """
        if not TRELLIS_AVAILABLE:
            raise ImportError("TRELLIS modules are not available")
            
        self.model_path = model_path
        self.base_output_dir = Path(base_output_dir)
        self.pipeline = None
        self.results_data: List[Dict] = []
        self.object_name_counter: Dict[str, int] = {}
        
        # 모델명 추출 (경로에서 마지막 부분)
        self.model_name = self._extract_model_name(model_path)
        
        # 현재 날짜
        self.current_date = datetime.now().strftime('%Y%m%d')
        
        # 출력 디렉토리 구조
        self.output_base = self.base_output_dir / "output" / self.model_name / self.current_date
        
        # 로깅 설정
        self._setup_logging()
    
    def _extract_model_name(self, model_path: str) -> str:
        """모델 경로에서 모델명 추출"""
        if '/' in model_path:
            # HuggingFace 형태 (microsoft/TRELLIS-text-xlarge) 또는 경로
            model_name = model_path.split('/')[-1]
        else:
            model_name = model_path
        
        # microsoft/ 접두사 제거
        if model_name.startswith('microsoft-'):
            model_name = model_name[10:]
        
        return model_name
    
    def _setup_logging(self):
        """로깅 설정"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('trellis_inference.log')
            ]
        )
    
    def create_default_yaml(self, filename: str = "default.yaml") -> None:
        """Create default YAML configuration file"""
        default_yaml_path = Path(filename)
        
        if not default_yaml_path.exists():
            default_config = {
                'prompts': [
                    "a red sports car",
                    "a wooden chair", 
                    "a blue coffee mug",
                    "a small house",
                    "a cute cat"
                ],
                'generation': {
                    'seed': 'random',
                    'sparse_structure_sampler_params': {
                        'steps': 12,
                        'cfg_strength': 7.5
                    },
                    'slat_sampler_params': {
                        'steps': 12,
                        'cfg_strength': 7.5
                    }
                },
                'output': {
                    'formats': ['glb', 'ply', 'mp4'],
                    'output_dir': './outputs'
                },
                'postprocessing': {
                    'simplify': 0.95,
                    'texture_size': 1024
                }
            }
            
            try:
                import yaml
                with open(default_yaml_path, 'w') as f:
                    yaml.dump(default_config, f, default_flow_style=False, indent=2)
                logging.info(f"📄 Created default.yaml configuration file")
            except ImportError:
                logging.error("❌ PyYAML not installed. Install with: pip install PyYAML")
    
    def load_yaml_config(self, yaml_filename: str) -> Optional[Dict]:
        """Load YAML configuration file"""
        try:
            import yaml
        except ImportError:
            logging.error("❌ PyYAML not installed. Install with: pip install PyYAML")
            return None
            
        try:
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
        logging.info(f"🔄 Loading TRELLIS pipeline from: {self.model_path}")
        try:
            # HuggingFace 모델명인지 로컬 경로인지 판단
            if self._is_huggingface_model(self.model_path):
                logging.info(f"📡 Loading HuggingFace model: {self.model_path}")
                self.pipeline = TrellisTextTo3DPipeline.from_pretrained(self.model_path)
            elif os.path.exists(self.model_path):
                logging.info(f"📁 Loading local model: {self.model_path}")
                self.pipeline = TrellisTextTo3DPipeline.from_pretrained(self.model_path)
            else:
                # 단순 모델명인 경우 microsoft/ 접두사 추가
                full_model_name = f"microsoft/{self.model_path}"
                logging.info(f"📡 Loading HuggingFace model: {full_model_name}")
                self.pipeline = TrellisTextTo3DPipeline.from_pretrained(full_model_name)
            
            # GPU 사용 가능시 GPU로 이동
            if torch.cuda.is_available():
                try:
                    self.pipeline.cuda()
                    logging.info("✅ Pipeline loaded on GPU successfully!")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logging.warning("⚠️ GPU out of memory, using CPU")
                        self.pipeline.cpu()
                    else:
                        raise
            else:
                logging.info("ℹ️ GPU not available, using CPU")
            
            # 모델 정보 출력
            self._print_model_info()
            
        except Exception as e:
            logging.error(f"❌ Pipeline loading failed: {e}")
            raise
    
    def _is_huggingface_model(self, model_path: str) -> bool:
        """Check if model path is a HuggingFace model name"""
        # HuggingFace 모델명 패턴: organization/model-name
        return '/' in model_path and not os.path.exists(model_path)
    
    def _print_model_info(self):
        """Print model information"""
        try:
            if hasattr(self.pipeline, 'models') and self.pipeline.models:
                logging.info("📊 Model components:")
                total_params = 0
                for name, model in self.pipeline.models.items():
                    if model is not None:
                        param_count = sum(p.numel() for p in model.parameters())
                        total_params += param_count
                        
                        # 양자화 상태 확인
                        is_quantized = any(
                            hasattr(m, '_packed_params') or 'quantized' in str(type(m)).lower()
                            for m in model.modules()
                        )
                        status = "🔧INT8" if is_quantized else "📏FP32"
                        
                        logging.info(f"  - {name}: {param_count/1e6:.1f}M params {status}")
                
                logging.info(f"📊 Total parameters: {total_params/1e6:.1f}M")
        except Exception as e:
            logging.warning(f"⚠️ Could not get model info: {e}")
    
    def generate_unique_name(self, base_prompt: str) -> str:
        """Generate unique object name based on prompt"""
        words = base_prompt.lower().split()
        clean_words = [word.strip('.,!?;:"()[]{}') for word in words if word.strip('.,!?;:"()[]{}')]
        english_words = [word for word in clean_words if word.isascii() and word.isalpha()]
        
        if not english_words:
            base_name = "object"
        else:
            base_name = "_".join(english_words[:3])
        
        if base_name in self.object_name_counter:
            self.object_name_counter[base_name] += 1
            return f"{base_name}_{self.object_name_counter[base_name]:02d}"
        else:
            self.object_name_counter[base_name] = 1
            return f"{base_name}_01"
    
    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize LLM model name for use in folder paths"""
        if not model_name:
            return "unknown_model"
        # Replace ':' with '_' and any other invalid characters
        return model_name.replace(':', '_')
    
    def process_batch_from_file(self, file_path: str, config: Dict, output_dir: str = "./outputs") -> None:
        """Process batch prompts from CSV or Excel file"""
        try:
            # 파일 확장자 확인
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()
            
            # 파일 형식에 따라 적절한 로더 사용
            if extension == '.csv':
                df = pd.read_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {extension}. CSV 또는 Excel 파일만 지원됩니다.")
            
            # CSV 컬럼 확인
            prompt_column = None
            if 'text_prompt' in df.columns:
                prompt_column = 'text_prompt'
            elif 'prompt' in df.columns:
                prompt_column = 'prompt'
            else:
                logging.error("❌ File must have either 'text_prompt' or 'prompt' column")
                return
            
            object_name_column = 'object_name' if 'object_name' in df.columns else None
            seed_column = 'seed' if 'seed' in df.columns else None
            models_column = 'llm_model' if 'llm_model' in df.columns else None
            
            # 파일 데이터를 딕셔너리 리스트로 변환
            file_data = []
            for _, row in df.iterrows():
                if pd.notna(row[prompt_column]):
                    item = {
                        'prompt': row[prompt_column],
                        'object_name': row[object_name_column] if object_name_column and pd.notna(row[object_name_column]) else None,
                        'seed': (
                            random.randint(0, 999999)
                            if seed_column and str(row[seed_column]).lower() == "random"
                            else int(row[seed_column]) if seed_column and pd.notna(row[seed_column])
                            else config.get('generation', {}).get('seed', "random")
                        ),
                        'llm_model': row[models_column] if models_column and pd.notna(row[models_column]) else None
                    }
                    file_data.append(item)
            
            logging.info(f"📊 Processing {len(file_data)} items from {extension.upper()} file")
            self._process_file_batch(file_data, config, output_dir)
            
        except Exception as e:
            logging.error(f"❌ File processing failed: {e}")
            raise
    
    def process_batch_from_yaml(self, yaml_path: str, output_dir: str = "./outputs") -> None:
        """Process batch prompts from YAML configuration"""
        config = self.load_yaml_config(yaml_path)
        if not config:
            return
        
        prompts = config.get('prompts', [])
        if not prompts:
            logging.error("❌ No prompts found in YAML config")
            return
        
        logging.info(f"📊 Processing {len(prompts)} prompts from YAML")
        self._process_yaml_batch(prompts, config, output_dir)
    
    def _process_file_batch(self, file_data: List[Dict], config: Dict, output_dir: str) -> None:
        """Process a batch of file data with individual settings"""
        # 사용자 지정 출력 디렉토리가 있으면 그것을 사용, 없으면 기본 구조 사용
        if output_dir != "./outputs":
            self.output_base = Path(output_dir) / self.model_name / self.current_date
        
        # 출력 디렉토리 생성
        self.output_base.mkdir(parents=True, exist_ok=True)

        generation_config = config.get('generation', {})
        output_config = config.get('output', {})
        postprocessing_config = config.get('postprocessing', {})
        
        formats = output_config.get('formats', ['glb'])
        
        for i, item in enumerate(file_data, 1):
            prompt = item['prompt']
            predefined_name = item['object_name']
            seed = item['seed']
            llm_model = item.get('llm_model')
            
            logging.info(f"\n🎯 [{i}/{len(file_data)}] Processing: '{prompt}'")
            if predefined_name:
                logging.info(f"📋 Object name: {predefined_name}")
            if llm_model:
                logging.info(f"🤖 LLM model: {llm_model}")
            logging.info(f"🎲 Seed: {seed}")
            
            try:
                # 개별 생성 설정
                individual_config = generation_config.copy()
                individual_config['seed'] = seed
                
                result = self._generate_single(
                    prompt=prompt,
                    predefined_name=predefined_name,
                    config=individual_config,
                    formats=formats,
                    postprocessing_config=postprocessing_config,
                    llm_model=llm_model
                )
                
                self.results_data.append(result)
                logging.info(f"✅ Completed: {result['object_name']}")
                
            except Exception as e:
                error_result = {
                    'prompt': prompt,
                    'object_name': predefined_name or 'error',
                    'seed': seed,
                    'model_name': self.model_name,
                    'llm_model': llm_model,
                    'generation_time': 0.0,
                    'total_time': 0.0,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.results_data.append(error_result)
                logging.error(f"❌ Failed: {prompt} - {e}")
        
        # Save results to CSV
        self._save_results_to_csv()
    
    def _process_yaml_batch(self, prompts: List[str], config: Dict, output_dir: str) -> None:
        """Process a batch of prompts from YAML"""
        # 사용자 지정 출력 디렉토리가 있으면 그것을 사용, 없으면 기본 구조 사용
        if output_dir != "./outputs":
            self.output_base = Path(output_dir) / self.model_name / self.current_date
        
        # 출력 디렉토리 생성
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        generation_config = config.get('generation', {})
        output_config = config.get('output', {})
        postprocessing_config = config.get('postprocessing', {})
        
        formats = output_config.get('formats', ['glb'])
        
        for i, prompt in enumerate(prompts, 1):
            logging.info(f"\n🎯 [{i}/{len(prompts)}] Processing: '{prompt}'")
            
            try:
                result = self._generate_single(
                    prompt=prompt,
                    predefined_name=None,
                    config=generation_config,
                    formats=formats,
                    postprocessing_config=postprocessing_config,
                    llm_model=None
                )
                
                self.results_data.append(result)
                logging.info(f"✅ Completed: {result['object_name']}")
                
            except Exception as e:
                error_result = {
                    'prompt': prompt,
                    'object_name': 'error',
                    'model_name': self.model_name,
                    'llm_model': None,
                    'generation_time': 0.0,
                    'total_time': 0.0,
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.results_data.append(error_result)
                logging.error(f"❌ Failed: {prompt} - {e}")
        
        # Save results to CSV
        self._save_results_to_csv()

    def _get_unique_filename(self, directory: Path, base_filename: str) -> str:
        """
        중복되는 파일명이 있으면 파일명 뒤에 001, 002, ... 형태로 숫자를 붙여서 유니크한 파일명 반환
        
        Args:
            directory: 파일이 저장될 디렉토리
            base_filename: 기본 파일명 (확장자 포함)
        
        Returns:
            유니크한 파일명 (확장자 포함)
        """
        filepath = directory / base_filename
        
        if not filepath.exists():
            return base_filename
        
        # 파일명과 확장자 분리
        name_part = base_filename.rsplit('.', 1)[0]  # 확장자 제거
        ext_part = '.' + base_filename.rsplit('.', 1)[1] if '.' in base_filename else ''  # 확장자
        
        # 중복되는 경우 숫자 접미사 추가
        counter = 1
        while True:
            new_filename = f"{name_part}_{counter:03d}{ext_part}"
            new_filepath = directory / new_filename
            
            if not new_filepath.exists():
                return new_filename
            
            counter += 1
            
            # 안전장치: 999개를 넘어가면 중단
            if counter > 999:
                # 타임스탬프를 추가하여 강제로 유니크하게 만듦
                timestamp = datetime.now().strftime('%H%M%S')
                new_filename = f"{name_part}_{timestamp}{ext_part}"
                return new_filename

    def _generate_single(self, prompt: str, predefined_name: Optional[str], config: Dict, formats: List[str], postprocessing_config: Dict, llm_model: Optional[str] = None) -> Dict:
        """Generate single 3D object"""
        start_time = time.time()
        
        # 객체 이름 결정
        if predefined_name:
            object_name = predefined_name
        else:
            object_name = self.generate_unique_name(prompt)
        
        # 시드 정보
        seed_val = config.get('seed', "random")
        if isinstance(seed_val, str) and seed_val.lower() == "random":
            seed = random.randint(0, 999999)
        else:
            seed = int(seed_val)
        
        # 개별 객체 디렉토리 생성: {base_output}/{llm_model}/{object_name}/ or {base_output}/{object_name}/
        if llm_model:
            sanitized_llm_model = self._sanitize_model_name(llm_model)
            object_dir = self.output_base / sanitized_llm_model / object_name
        else:
            object_dir = self.output_base / object_name
        print(f'LLM Model: {llm_model}, target object_dir: {object_dir}')
        object_dir.mkdir(parents=True, exist_ok=True)
        
        # Generation timing
        gen_start = time.time()
        
        try:
            outputs = self.pipeline.run(
                prompt,
                seed=seed,
                sparse_structure_sampler_params=config.get('sparse_structure_sampler_params', {}),
                slat_sampler_params=config.get('slat_sampler_params', {})
            )
        except Exception as e:
            logging.error(f"❌ Pipeline execution failed: {e}")
            raise
            
        generation_time = time.time() - gen_start
        
        # Render different video types
        render_start = time.time()
        video_gs = None
        video_rf = None
        video_mesh = None
        
        try:
            if 'mp4' in formats:
                video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
                video_rf = render_utils.render_video(outputs['radiance_field'][0])['color']
                video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
        except Exception as e:
            logging.warning(f"⚠️ Video rendering failed: {e}")
        
        render_time = time.time() - render_start
        
        # Save outputs in requested formats
        saved_files = []
        save_start = time.time()
        
        try:
            # GLB 파일: {object_name}_{model_name}_{llm_model}_{seed}.glb (중복 시 숫자 추가)
            if 'glb' in formats:
                if llm_model:
                    sanitized_llm_model = self._sanitize_model_name(llm_model)
                    base_filename = f"{object_name}_{self.model_name}_{sanitized_llm_model}_{seed}.glb"
                else:
                    base_filename = f"{object_name}_{self.model_name}_{seed}.glb"
                glb_filename = self._get_unique_filename(object_dir, base_filename)
                glb_path = object_dir / glb_filename
                
                glb = postprocessing_utils.to_glb(
                    outputs['gaussian'][0],
                    outputs['mesh'][0],
                    simplify=postprocessing_config.get('simplify', 0.95),
                    texture_size=postprocessing_config.get('texture_size', 1024)
                )
                glb.export(str(glb_path))
                saved_files.append(str(glb_path))
                logging.info(f"💾 GLB saved: {glb_filename}")
            
            # PLY 파일: {object_name}_{model_name}_{llm_model}_{seed}.ply (중복 시 숫자 추가)
            if 'ply' in formats:
                if llm_model:
                    sanitized_llm_model = self._sanitize_model_name(llm_model)
                    base_filename = f"{object_name}_{self.model_name}_{sanitized_llm_model}_{seed}.ply"
                else:
                    base_filename = f"{object_name}_{self.model_name}_{seed}.ply"
                ply_filename = self._get_unique_filename(object_dir, base_filename)
                ply_path = object_dir / ply_filename
                
                outputs['gaussian'][0].save_ply(str(ply_path))
                saved_files.append(str(ply_path))
                logging.info(f"💾 PLY saved: {ply_filename}")
            
            # MP4 파일들: {object_name}_{model_name}_{llm_model}_{seed}_gs/rf/mesh.mp4 (중복 시 숫자 추가)
            if 'mp4' in formats:
                if video_gs is not None:
                    if llm_model:
                        sanitized_llm_model = self._sanitize_model_name(llm_model)
                        base_filename = f"{object_name}_{self.model_name}_{sanitized_llm_model}_{seed}_gs.mp4"
                    else:
                        base_filename = f"{object_name}_{self.model_name}_{seed}_gs.mp4"
                    gs_filename = self._get_unique_filename(object_dir, base_filename)
                    gs_path = object_dir / gs_filename
                    imageio.mimsave(str(gs_path), video_gs, fps=30)
                    saved_files.append(str(gs_path))
                    logging.info(f"💾 GS video saved: {gs_filename}")
                
                if video_rf is not None:
                    if llm_model:
                        sanitized_llm_model = self._sanitize_model_name(llm_model)
                        base_filename = f"{object_name}_{self.model_name}_{sanitized_llm_model}_{seed}_rf.mp4"
                    else:
                        base_filename = f"{object_name}_{self.model_name}_{seed}_rf.mp4"
                    rf_filename = self._get_unique_filename(object_dir, base_filename)
                    rf_path = object_dir / rf_filename
                    imageio.mimsave(str(rf_path), video_rf, fps=30)
                    saved_files.append(str(rf_path))
                    logging.info(f"💾 RF video saved: {rf_filename}")
                
                if video_mesh is not None:
                    if llm_model:
                        sanitized_llm_model = self._sanitize_model_name(llm_model)
                        base_filename = f"{object_name}_{self.model_name}_{sanitized_llm_model}_{seed}_mesh.mp4"
                    else:
                        base_filename = f"{object_name}_{self.model_name}_{seed}_mesh.mp4"
                    mesh_filename = self._get_unique_filename(object_dir, base_filename)
                    mesh_path = object_dir / mesh_filename
                    imageio.mimsave(str(mesh_path), video_mesh, fps=30)
                    saved_files.append(str(mesh_path))
                    logging.info(f"💾 Mesh video saved: {mesh_filename}")
            
            # 썸네일 생성: {object_name}_{model_name}_{llm_model}_{seed}_gs_{sec}s.jpg (중복 시 숫자 추가)
            if 'jpg' in formats or video_gs is not None:
                try:
                    frame_times = [4, 5, 6, 10]  # seconds
                    fps = 30  # same as render

                    from PIL import Image
                    for sec in frame_times:
                        frame_idx = sec * fps
                        if video_gs is not None and len(video_gs) > frame_idx:
                            if llm_model:
                                sanitized_llm_model = self._sanitize_model_name(llm_model)
                                base_filename = f"{object_name}_{self.model_name}_{sanitized_llm_model}_{seed}_gs_{sec:03d}s.jpg"
                            else:
                                base_filename = f"{object_name}_{self.model_name}_{seed}_gs_{sec:03d}s.jpg"
                            thumbnail_filename = self._get_unique_filename(object_dir, base_filename)
                            thumbnail_path = object_dir / thumbnail_filename
                            
                            thumbnail_img = Image.fromarray(video_gs[frame_idx])
                            thumbnail_img.save(str(thumbnail_path), "JPEG", quality=90)
                            saved_files.append(str(thumbnail_path))
                            logging.info(f"💾 Thumbnail saved: {thumbnail_filename}")
                        else:
                            logging.warning(f"⚠️ Frame {frame_idx} for {sec}s not available in video_gs")
                except Exception as e:
                    logging.warning(f"⚠️ Thumbnail generation failed: {e}")
                    
        except Exception as e:
            logging.error(f"❌ File saving failed: {e}")
            logging.error(f"   Tried to save to: {object_dir}")
            raise
        
        save_time = time.time() - save_start
        total_time = time.time() - start_time
        
        return {
            'prompt': prompt,
            'object_name': object_name,
            'seed': seed,
            'model_name': self.model_name,
            'llm_model': llm_model,
            'generation_time': round(generation_time, 2),
            'render_time': round(render_time, 2),
            'save_time': round(save_time, 2),
            'total_time': round(total_time, 2),
            'success': True,
            'saved_files': saved_files,
            'save_path': str(object_dir),
            'timestamp': datetime.now().isoformat()
        }


    def _save_results_to_csv(self) -> None:
        """Save results to CSV file with specified naming format"""
        if not self.results_data:
            return
        
        # CSV 파일명: results_{model_name}_{current_date}_{time}.csv
        current_time = datetime.now().strftime('%H%M%S')
        csv_filename = f"results_{self.model_name}_{self.current_date}_{current_time}.csv"
        csv_path = self.output_base / csv_filename
        
        results_df = pd.DataFrame(self.results_data)
        results_df.to_csv(csv_path, index=False)
        
        logging.info(f"📊 Results saved to: {csv_path}")
        
        # Print summary
        successful = sum(1 for r in self.results_data if r.get('success', False))
        total = len(self.results_data)
        avg_time = sum(r.get('generation_time', 0) for r in self.results_data if r.get('success', False)) / max(successful, 1)
        
        logging.info(f"✅ Summary: {successful}/{total} successful, avg time: {avg_time:.1f}s")
        logging.info(f"📁 All files saved in: {self.output_base}")


def main():
    parser = argparse.ArgumentParser(
        description="TRELLIS Text-to-3D Batch Inference Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default config
  python trellis_inference.py --create_default
  
  # Use default HuggingFace model with YAML
  python trellis_inference.py --config default.yaml
  
  # Use specific model with custom base path
  python trellis_inference.py --model_path TRELLIS-text-large --config default.yaml --base_output /custom/path
  
  # Process CSV/Excel input with custom output
  python trellis_inference.py --csv prompts.csv --output /mnt/nas/tmp/nayeon --base_output /mnt/nas/tmp/nayeon
  python trellis_inference.py --csv prompts.xlsx --output /mnt/nas/tmp/nayeon --base_output /mnt/nas/tmp/nayeon
        """
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='microsoft/TRELLIS-text-xlarge',
        help='Model path (local path or HuggingFace model name)'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        help='YAML configuration file path'
    )
    
    parser.add_argument(
        '--csv', 
        type=str, 
        help='CSV or Excel file path with prompts (.csv, .xlsx, .xls)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='./outputs',
        help='Output directory for this run'
    )
    
    parser.add_argument(
        '--base_output', 
        type=str, 
        default='/mnt/nas/tmp/nayeon',
        help='Base output directory for structured file organization'
    )
    
    parser.add_argument(
        '--create_default', 
        action='store_true',
        help='Create default.yaml configuration file and exit'
    )
    
    args = parser.parse_args()
    print("args: ", args)

    # Create default config and exit
    if args.create_default:
        try:
            manager = TrellisInferenceManager(base_output_dir=args.base_output)
            manager.create_default_yaml()
            print("✅ Created default.yaml configuration file")
        except Exception as e:
            print(f"❌ Error creating default config: {e}")
        return
    
    # Validate input arguments
    if not args.config and not args.csv:
        print("❌ Error: Either --config or --csv must be specified")
        print("💡 Use --create_default to create a default configuration file")
        return
    
    try:
        # Initialize manager with specified model path and base output
        print("Initialize model path: ", args.model_path)
        manager = TrellisInferenceManager(model_path=args.model_path, base_output_dir=args.base_output)
        print("Load pipeline")
        manager.load_pipeline()
        
        # Process based on input type
        if args.csv:
            # Create minimal config for CSV processing
            config = {
                'generation': {'seed': "random"},
                'output': {'formats': ['glb', 'ply', 'mp4', 'jpg']},
                'postprocessing': {'simplify': 0.95, 'texture_size': 1024}
            }
            manager.process_batch_from_file(args.csv, config, args.output)
        elif args.config:
            manager.process_batch_from_yaml(args.config, args.output)
        
        print(f"\n🎉 Batch processing completed!")
        print(f"📁 Files saved in: {manager.output_base}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


    return 0


if __name__ == "__main__":
    exit(main())