"""
TRELLIS 양자화 실험을 위한 공통 평가 지표 모듈 (TRELLIS 논문 방법론 적용)

이 모듈은 TRELLIS 논문의 평가 방법론에 따라 다음 지표를 제공합니다:

1. CLIP Score (TRELLIS 논문 방식):
   - 각 생성된 3D 자산에 대해 8개 이미지 렌더링 (Yaw 45도 간격, Pitch 30°, 반경 2)
   - 렌더링된 이미지와 텍스트 프롬프트의 CLIP 코사인 유사도 계산
   - 평균 유사도에 100을 곱하여 점수화

2. Fréchet Distance (FD) with DINOv2:
   - 참조 데이터셋과 생성된 자산들의 DINOv2 특징 분포 비교
   - 실제 데이터와 생성 데이터의 품질 및 다양성 평가

3. 효율성 지표: 파라미터 수, 모델 크기, GPU 메모리, 추론 시간

사용법:
    evaluator = MetricsEvaluator()
    clip_score = evaluator.compute_clip_score_trellis_paper(text_prompts, generated_3d_outputs)
    fd_score = evaluator.compute_frechet_distance_trellis_paper(reference_assets, generated_assets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
import tempfile
from typing import List, Dict, Tuple, Any, Union
from PIL import Image
from transformers import CLIPTextModel, CLIPVisionModel, CLIPProcessor, AutoTokenizer
from torchvision import transforms
from scipy.linalg import sqrtm
from pathlib import Path
from .trellis_render_utils import (
    render_pipeline_output_trellis_paper,
    load_rendered_images,
    create_trellis_paper_views
)


class MetricsEvaluator:
    """TRELLIS 양자화 실험을 위한 공통 평가 지표 클래스"""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델들을 lazy loading으로 초기화
        self._clip_text_model = None
        self._clip_vision_model = None
        self._clip_processor = None
        self._clip_tokenizer = None
        self._dinov2_model = None
        self._dinov2_transform = None
        # 렌더링 유틸리티는 함수로 직접 사용
        
        print(f"📊 MetricsEvaluator 초기화됨 (device: {self.device})")
    
    def _init_clip_model(self, clip_model_name: str = "openai/clip-vit-large-patch14"):
        """CLIP 모델 초기화 (텍스트와 비전 모델 모두)"""
        if self._clip_text_model is None or self._clip_vision_model is None:
            print(f"🔧 CLIP 모델 로딩 중: {clip_model_name}")
            
            # CLIP 텍스트 및 비전 모델 로드
            self._clip_text_model = CLIPTextModel.from_pretrained(clip_model_name)
            self._clip_vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
            self._clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self._clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_name)
            
            self._clip_text_model.eval()
            self._clip_vision_model.eval()
            self._clip_text_model = self._clip_text_model.to(self.device)
            self._clip_vision_model = self._clip_vision_model.to(self.device)
            
            print("✅ CLIP 모델 로딩 완료")
    
    def _init_dinov2_model(self, model_name: str = "dinov2_vitl14"):
        """DINOv2 모델 초기화 (extract_feature.py 방식 참고)"""
        if self._dinov2_model is None:
            print(f"🔧 DINOv2 모델 로딩 중: {model_name}")
            
            # extract_feature.py의 방식을 참고
            self._dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name)
            self._dinov2_model.eval()
            self._dinov2_model = self._dinov2_model.to(self.device)
            
            # DINOv2용 전처리 (extract_feature.py 참고)
            self._dinov2_transform = transforms.Compose([
                transforms.Resize((518, 518)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ DINOv2 모델 로딩 완료")
    
    # 렌더링은 trellis_render_utils 함수 직접 사용
    
    def encode_text_clip(self, text_list: List[str]) -> torch.Tensor:
        """텍스트를 CLIP으로 인코딩 (정규화된 특징 벡터 반환)"""
        self._init_clip_model()
        
        encoding = self._clip_tokenizer(
            text_list, 
            max_length=77, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        tokens = encoding['input_ids'].to(self.device)
        
        with torch.no_grad():
            text_features = self._clip_text_model(input_ids=tokens).pooler_output
            # L2 정규화
            text_features = F.normalize(text_features, p=2, dim=-1)
        
        return text_features
    
    def encode_images_clip(self, images: List[Image.Image]) -> torch.Tensor:
        """이미지들을 CLIP으로 인코딩 (정규화된 특징 벡터 반환)"""
        self._init_clip_model()
        
        # 이미지 전처리
        inputs = self._clip_processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            image_features = self._clip_vision_model(pixel_values=pixel_values).pooler_output
            # L2 정규화
            image_features = F.normalize(image_features, p=2, dim=-1)
        
        return image_features
    
    def extract_dinov2_features(self, images: List[Union[torch.Tensor, Image.Image]]) -> torch.Tensor:
        """이미지에서 DINOv2 특징 추출"""
        self._init_dinov2_model()
        
        if not images:
            raise ValueError("이미지 리스트가 비어있습니다.")
        
        # 이미지 전처리
        processed_images = []
        for img in images:
            if isinstance(img, torch.Tensor):
                # 이미 tensor인 경우
                if img.dim() == 4:  # batch dimension이 있는 경우
                    img = img.squeeze(0)
                if img.shape[0] == 3:  # CHW 형식
                    img = transforms.ToPILImage()(img)
                else:  # HWC 형식
                    img = transforms.ToPILImage()(img.permute(2, 0, 1))
            
            # PIL Image로 변환 후 전처리
            if isinstance(img, Image.Image):
                processed_img = self._dinov2_transform(img)
                processed_images.append(processed_img)
        
        if not processed_images:
            raise ValueError("처리할 수 있는 이미지가 없습니다.")
        
        # 배치로 묶기
        batch_tensor = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            features = self._dinov2_model(batch_tensor)
        
        return features
    
    def compute_clip_score_trellis_paper(self, text_prompts: List[str], generated_3d_assets: List[Any], 
                                        temp_dir: str = None) -> Dict[str, float]:
        """
        TRELLIS 논문 방식의 CLIP Score 계산
        
        측정 방식:
        1. 각 생성된 3D 자산에 대해 8개의 이미지를 렌더링
           - 카메라 설정: Yaw 45도 간격(0°, 45°, ..., 315°), Pitch 30°, 반경 2
        2. 렌더링된 이미지와 텍스트 프롬프트의 CLIP 특징 추출
        3. 코사인 유사도 계산 후 평균하고 100을 곱해서 점수화
        
        Args:
            text_prompts: 텍스트 프롬프트 리스트
            generated_3d_assets: 생성된 3D 자산들 (파이프라인 출력)
            temp_dir: 임시 디렉토리 (렌더링 결과 저장)
        
        Returns:
            Dict: CLIP 점수 통계
        """
        try:
            print("📐 TRELLIS 논문 방식 CLIP Score 계산 중...")
            
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            
            self._init_renderer()
            
            all_clip_scores = []
            
            for i, (prompt, asset) in enumerate(zip(text_prompts, generated_3d_assets)):
                print(f"  자산 {i+1}/{len(text_prompts)} 처리 중...")
                
                try:
                    # 1. 3D 자산을 8개 뷰로 렌더링
                    asset_output_dir = f"{temp_dir}/asset_{i}"
                    rendered_images = render_pipeline_output_trellis_paper(
                        asset, asset_output_dir, f"asset_{i}"
                    )
                    
                    if len(rendered_images) == 0:
                        print(f"    ⚠️ 자산 {i+1} 렌더링 실패")
                        continue
                    
                    # 이미지 로드
                    images = load_rendered_images(rendered_images)
                    
                    if len(images) == 0:
                        print(f"    ⚠️ 자산 {i+1} 이미지 로드 실패")
                        continue
                    
                    # 2. CLIP 특징 추출
                    text_features = self.encode_text_clip([prompt])  # [1, feature_dim]
                    image_features = self.encode_images_clip(images)  # [N_views, feature_dim]
                    
                    # 3. 코사인 유사도 계산
                    similarities = torch.matmul(image_features, text_features.T)  # [N_views, 1]
                    similarities = similarities.squeeze(-1)  # [N_views]
                    
                    # 4. 모든 뷰의 유사도 평균 계산
                    avg_similarity = similarities.mean().item()
                    
                    # 5. TRELLIS 논문 방식: 100을 곱해서 점수화
                    clip_score = avg_similarity * 100
                    all_clip_scores.append(clip_score)
                    
                    print(f"    자산 {i+1} CLIP 점수: {clip_score:.2f} ({len(images)}개 뷰)")
                    
                except Exception as e:
                    print(f"    ⚠️ 자산 {i+1} 처리 실패: {e}")
                    continue
            
            if len(all_clip_scores) == 0:
                print("❌ 모든 자산 처리 실패")
                return {
                    'clip_score_mean': 0.0,
                    'clip_score_std': 0.0,
                    'clip_score_min': 0.0,
                    'clip_score_max': 0.0,
                    'num_samples': 0
                }
            
            # 통계 계산
            results = {
                'clip_score_mean': np.mean(all_clip_scores),
                'clip_score_std': np.std(all_clip_scores),
                'clip_score_min': np.min(all_clip_scores),
                'clip_score_max': np.max(all_clip_scores),
                'num_samples': len(all_clip_scores)
            }
            
            print(f"✅ CLIP Score 계산 완료: 평균 {results['clip_score_mean']:.2f} ({results['num_samples']}개 샘플)")
            return results
            
        except Exception as e:
            print(f"❌ CLIP Score 계산 실패: {e}")
            return {
                'clip_score_mean': 0.0,
                'clip_score_std': 0.0,
                'clip_score_min': 0.0,
                'clip_score_max': 0.0,
                'num_samples': 0
            }
    
    def compute_frechet_distance(self, real_features: torch.Tensor, generated_features: torch.Tensor) -> float:
        """
        Fréchet Distance 계산
        
        Args:
            real_features: 실제 이미지의 DINOv2 특징
            generated_features: 생성된 이미지의 DINOv2 특징
            
        Returns:
            float: Fréchet Distance 값
        """
        try:
            print("📏 Fréchet Distance 계산 중...")
            
            # CPU로 이동
            real_features = real_features.cpu().numpy()
            generated_features = generated_features.cpu().numpy()
            
            # 평균과 공분산 계산
            mu_real = np.mean(real_features, axis=0)
            mu_gen = np.mean(generated_features, axis=0)
            
            sigma_real = np.cov(real_features, rowvar=False)
            sigma_gen = np.cov(generated_features, rowvar=False)
            
            # Fréchet Distance 계산
            diff = mu_real - mu_gen
            covmean = sqrtm(sigma_real.dot(sigma_gen))
            
            # 수치적 안정성을 위한 처리
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            
            fd_score = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
            
            return float(fd_score)
            
        except Exception as e:
            print(f"⚠️ Fréchet Distance 계산 실패: {e}")
            return 100.0  # 높은 값 반환 (나쁜 품질 의미)
    
    def compute_frechet_distance_trellis_paper(self, reference_3d_assets: List[Any], 
                                              generated_3d_assets: List[Any],
                                              temp_dir: str = None) -> float:
        """
        TRELLIS 논문 방식의 Fréchet Distance 계산
        
        측정 방식:
        1. 참조 데이터셋과 생성된 3D 자산들을 각각 렌더링
        2. DINOv2로 특징 추출 
        3. 두 분포 간의 Fréchet Distance 계산
        
        Args:
            reference_3d_assets: 참조 3D 자산들 (실제 데이터)
            generated_3d_assets: 생성된 3D 자산들
            temp_dir: 임시 디렉토리
            
        Returns:
            float: Fréchet Distance 값
        """
        try:
            print("📏 TRELLIS 논문 방식 Fréchet Distance 계산 중...")
            
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp()
            
            self._init_renderer()
            
            # 1. 참조 데이터 렌더링 및 특징 추출
            print("  참조 데이터 처리 중...")
            reference_images = []
            for i, asset in enumerate(reference_3d_assets):
                asset_output_dir = f"{temp_dir}/ref_{i}"
                rendered_images = render_pipeline_output_trellis_paper(
                    asset, asset_output_dir, f"ref_{i}"
                )
                images = load_rendered_images(rendered_images)
                reference_images.extend(images)
            
            # 2. 생성된 데이터 렌더링 및 특징 추출
            print("  생성된 데이터 처리 중...")
            generated_images = []
            for i, asset in enumerate(generated_3d_assets):
                asset_output_dir = f"{temp_dir}/gen_{i}"
                rendered_images = render_pipeline_output_trellis_paper(
                    asset, asset_output_dir, f"ref_{i}"
                )
                images = load_rendered_images(rendered_images)
                generated_images.extend(images)
            
            # 3. DINOv2 특징 추출
            if len(reference_images) == 0 or len(generated_images) == 0:
                print("❌ 렌더링된 이미지가 없음")
                return 100.0
                
            reference_features = self.extract_dinov2_features(reference_images)
            generated_features = self.extract_dinov2_features(generated_images)
            
            # 4. Fréchet Distance 계산
            fd_score = self.compute_frechet_distance(reference_features, generated_features)
            
            print(f"✅ Fréchet Distance 계산 완료: {fd_score:.2f}")
            print(f"  참조 이미지: {len(reference_images)}개, 생성 이미지: {len(generated_images)}개")
            
            return fd_score
            
        except Exception as e:
            print(f"❌ Fréchet Distance 계산 실패: {e}")
            return 100.0
    
    def compute_frechet_distance_from_images(self, real_images: List[Image.Image], 
                                           generated_images: List[Image.Image]) -> float:
        """
        이미지로부터 Fréchet Distance 계산 (기존 방식 유지)
        
        Args:
            real_images: 실제 이미지 리스트  
            generated_images: 생성된 이미지 리스트
            
        Returns:
            float: Fréchet Distance 값
        """
        try:
            # DINOv2 특징 추출
            real_features = self.extract_dinov2_features(real_images)
            generated_features = self.extract_dinov2_features(generated_images)
            
            # FD 계산
            return self.compute_frechet_distance(real_features, generated_features)
            
        except Exception as e:
            print(f"⚠️ 이미지 기반 Fréchet Distance 계산 실패: {e}")
            return 100.0
    
    def compute_efficiency_metrics(self, pipeline, model_name: str) -> Dict[str, float]:
        """효율성 지표 계산"""
        print(f"📊 {model_name} 효율성 지표 계산 중...")
        
        metrics = {}
        
        try:
            # 1. 파라미터 수 계산
            total_params = 0
            if hasattr(pipeline, 'models') and pipeline.models:
                for name, module in pipeline.models.items():
                    if module is not None:
                        params = sum(p.numel() for p in module.parameters())
                        total_params += params
            
            metrics['parameters_M'] = total_params / 1e6
            
            # 2. 모델 크기 계산 (MB)
            total_size = 0
            if hasattr(pipeline, 'models') and pipeline.models:
                for name, module in pipeline.models.items():
                    if module is not None:
                        size = sum(p.numel() * p.element_size() for p in module.parameters())
                        total_size += size
            
            metrics['model_size_MB'] = total_size / (1024 * 1024)
            
            # 3. GPU 메모리 사용량 측정
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                try:
                    # 더미 입력으로 메모리 사용량 측정
                    dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
                    
                    with torch.no_grad():
                        pipeline(dummy_image, num_inference_steps=1)
                    
                    metrics['gpu_memory_MB'] = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    
                except Exception as e:
                    print(f"    ⚠️ GPU 메모리 측정 실패: {e}")
                    metrics['gpu_memory_MB'] = 0
            else:
                metrics['gpu_memory_MB'] = 0
            
            # 4. 추론 시간 측정
            inference_times = []
            num_runs = 3
            
            try:
                dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
                
                # 워밍업
                with torch.no_grad():
                    pipeline(dummy_image, num_inference_steps=1)
                
                # 실제 측정
                for _ in range(num_runs):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    
                    with torch.no_grad():
                        pipeline(dummy_image, num_inference_steps=1)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    inference_times.append((end_time - start_time) * 1000)  # ms
                
                metrics['inference_time_ms'] = np.mean(inference_times)
                metrics['inference_time_std'] = np.std(inference_times)
                
            except Exception as e:
                print(f"    ⚠️ 추론 시간 측정 실패: {e}")
                metrics['inference_time_ms'] = 0
                metrics['inference_time_std'] = 0
            
            return metrics
            
        except Exception as e:
            print(f"❌ 효율성 지표 계산 실패: {e}")
            return {
                'parameters_M': 0,
                'model_size_MB': 0,
                'gpu_memory_MB': 0,
                'inference_time_ms': 0,
                'inference_time_std': 0
            }
    
    def evaluate_pipeline_quality_trellis_paper(self, pipeline, text_prompts: List[str], 
                                               reference_assets: List[Any] = None,
                                               num_samples: int = 5, temp_dir: str = None) -> Dict[str, float]:
        """
        TRELLIS 논문 방식의 파이프라인 품질 평가
        
        Args:
            pipeline: TRELLIS 파이프라인
            text_prompts: 평가용 텍스트 프롬프트들
            reference_assets: 참조 3D 자산들 (FD 계산용)
            num_samples: 샘플 수
            temp_dir: 임시 디렉토리
            
        Returns:
            Dict: 품질 지표들
        """
        print(f"🎯 TRELLIS 논문 방식 파이프라인 품질 평가 중 (샘플: {num_samples}개)...")
        
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        
        try:
            generated_assets = []
            used_prompts = []
            
            # 각 프롬프트에 대해 생성
            for i, prompt in enumerate(text_prompts[:num_samples]):
                print(f"  샘플 {i+1}/{min(num_samples, len(text_prompts))}: '{prompt[:50]}...'")
                
                try:
                    dummy_image = torch.randn(1, 3, 512, 512).to(self.device)
                    
                    with torch.no_grad():
                        output = pipeline(dummy_image, num_inference_steps=5)
                    
                    generated_assets.append(output)
                    used_prompts.append(prompt)
                    
                except Exception as e:
                    print(f"    ⚠️ 샘플 {i+1} 생성 실패: {e}")
                    continue
            
            if len(generated_assets) == 0:
                print("❌ 생성된 자산이 없음")
                return {
                    'clip_score_mean': 0.0,
                    'clip_score_std': 0.0,
                    'frechet_distance': 100.0
                }
            
            results = {}
            
            # 1. TRELLIS 논문 방식 CLIP Score 계산
            try:
                clip_results = self.compute_clip_score_trellis_paper(
                    used_prompts, generated_assets, temp_dir
                )
                results.update(clip_results)
            except Exception as e:
                print(f"⚠️ CLIP Score 계산 건너뛰기: {e}")
                results.update({
                    'clip_score_mean': 0.0,
                    'clip_score_std': 0.0
                })
            
            # 2. TRELLIS 논문 방식 Fréchet Distance 계산
            if reference_assets and len(reference_assets) > 0:
                try:
                    fd_score = self.compute_frechet_distance_trellis_paper(
                        reference_assets, generated_assets, temp_dir
                    )
                    results['frechet_distance'] = fd_score
                except Exception as e:
                    print(f"⚠️ FD 계산 건너뛰기: {e}")
                    results['frechet_distance'] = 100.0
            else:
                print("⚠️ 참조 자산이 없어 FD 계산 건너뛰기")
                results['frechet_distance'] = 100.0
            
            return results
            
        except Exception as e:
            print(f"❌ 파이프라인 품질 평가 실패: {e}")
            return {
                'clip_score_mean': 0.0,
                'clip_score_std': 0.0,
                'frechet_distance': 100.0
            }
    
    def evaluate_pipeline_quality(self, pipeline, text_prompts: List[str], 
                                num_samples: int = 5) -> Dict[str, float]:
        """
        파이프라인의 전체 품질 평가 (기존 방식, 호환성 유지)
        
        Args:
            pipeline: TRELLIS 파이프라인
            text_prompts: 평가용 텍스트 프롬프트들
            num_samples: 샘플 수
            
        Returns:
            Dict: 품질 지표들
        """
        # TRELLIS 논문 방식 사용
        return self.evaluate_pipeline_quality_trellis_paper(
            pipeline, text_prompts, num_samples=num_samples
        )
    
    def cleanup(self):
        """메모리 정리"""
        print("🧹 MetricsEvaluator 메모리 정리 중...")
        
        if self._clip_text_model is not None:
            del self._clip_text_model
            self._clip_text_model = None
        
        if self._clip_vision_model is not None:
            del self._clip_vision_model
            self._clip_vision_model = None
        
        if self._clip_processor is not None:
            del self._clip_processor
            self._clip_processor = None
        
        if self._clip_tokenizer is not None:
            del self._clip_tokenizer
            self._clip_tokenizer = None
        
        if self._dinov2_model is not None:
            del self._dinov2_model
            self._dinov2_model = None
        
        if self._dinov2_transform is not None:
            del self._dinov2_transform
            self._dinov2_transform = None
        
        # 렌더링 유틸리티는 함수이므로 정리 불필요
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ 메모리 정리 완료")


# 전역 평가기 인스턴스 (싱글톤 패턴)
_global_evaluator = None

def get_metrics_evaluator(device: str = None) -> MetricsEvaluator:
    """전역 MetricsEvaluator 인스턴스 반환 (싱글톤)"""
    global _global_evaluator
    
    if _global_evaluator is None:
        _global_evaluator = MetricsEvaluator(device)
    
    return _global_evaluator


def cleanup_global_evaluator():
    """전역 평가기 정리"""
    global _global_evaluator
    
    if _global_evaluator is not None:
        _global_evaluator.cleanup()
        _global_evaluator = None