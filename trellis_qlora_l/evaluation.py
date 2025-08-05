"""
모델 평가 유틸리티

주요 기능:
- CLIP Score 계산
- FID, KID 등 생성 품질 평가
- Chamfer Distance 계산
- 3D 모델 품질 메트릭
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️ CLIP 라이브러리가 없습니다. pip install clip-by-openai")

try:
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.structures import Pointclouds
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("⚠️ PyTorch3D가 없습니다. 3D 메트릭을 사용할 수 없습니다.")

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("⚠️ TorchMetrics가 없습니다. FID/KID 계산을 사용할 수 없습니다.")


class TRELLISEvaluator:
    """TRELLIS 모델 평가기"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clip_model = None
        self.clip_preprocess = None
        
        # CLIP 모델 로드
        if CLIP_AVAILABLE:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
            self.clip_model.eval()
        
        # FID/KID 계산기
        if TORCHMETRICS_AVAILABLE:
            self.fid = FrechetInceptionDistance(feature=2048).to(device)
            self.kid = KernelInceptionDistance(subset_size=100).to(device)
    
    def evaluate_model(self, model, test_prompts: List[str], config) -> Dict[str, float]:
        """모델 전체 평가"""
        print("📊 모델 평가 시작...")
        
        results = {}
        generated_samples = []
        
        # 샘플 생성
        for i, prompt in enumerate(test_prompts):
            print(f"🔄 생성 중 ({i+1}/{len(test_prompts)}): {prompt}")
            
            try:
                # 3D 생성
                outputs = model.run(prompt, seed=42 + i)
                generated_samples.append({
                    'prompt': prompt,
                    'outputs': outputs,
                    'index': i
                })
            except Exception as e:
                print(f"❌ 생성 실패: {prompt} - {e}")
                continue
        
        if not generated_samples:
            print("❌ 생성된 샘플이 없습니다.")
            return {}
        
        print(f"✅ {len(generated_samples)}개 샘플 생성 완료")
        
        # 평가 메트릭 계산
        if CLIP_AVAILABLE:
            results['clip_score'] = self.calculate_clip_score(generated_samples)
        
        if PYTORCH3D_AVAILABLE:
            results['chamfer_distance'] = self.calculate_chamfer_distance(generated_samples)
        
        # 렌더링 품질 평가 (FID/KID)
        if TORCHMETRICS_AVAILABLE:
            fid_score, kid_score = self.calculate_image_metrics(generated_samples)
            results['fid_score'] = fid_score
            results['kid_score'] = kid_score
        
        # 추론 시간 측정
        results['inference_time'] = self.measure_inference_time(model, test_prompts[:5])
        
        print("📊 평가 완료!")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")
        
        return results
    
    def calculate_clip_score(self, samples: List[Dict]) -> float:
        """CLIP Score 계산"""
        if not CLIP_AVAILABLE or not self.clip_model:
            return 0.0
        
        print("🔍 CLIP Score 계산 중...")
        clip_scores = []
        
        for sample in samples:
            try:
                prompt = sample['prompt']
                outputs = sample['outputs']
                
                # 3D에서 2D 렌더링 생성 (간단한 예시)
                rendered_images = self.render_3d_to_2d(outputs)
                
                if rendered_images is None:
                    continue
                
                # 텍스트 인코딩
                text_tokens = clip.tokenize([prompt]).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 이미지 인코딩 및 유사도 계산
                image_scores = []
                for img in rendered_images:
                    img_preprocessed = self.clip_preprocess(img).unsqueeze(0).to(self.device)
                    img_features = self.clip_model.encode_image(img_preprocessed)
                    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                    
                    # 코사인 유사도
                    similarity = torch.cosine_similarity(text_features, img_features)
                    image_scores.append(similarity.item())
                
                if image_scores:
                    clip_scores.append(max(image_scores))  # 최고 점수 사용
                
            except Exception as e:
                print(f"⚠️ CLIP Score 계산 오류: {e}")
                continue
        
        avg_clip_score = np.mean(clip_scores) if clip_scores else 0.0
        print(f"📊 CLIP Score: {avg_clip_score:.4f}")
        return avg_clip_score
    
    def calculate_chamfer_distance(self, samples: List[Dict]) -> float:
        """Chamfer Distance 계산"""
        if not PYTORCH3D_AVAILABLE:
            return 0.0
        
        print("🔍 Chamfer Distance 계산 중...")
        cd_scores = []
        
        for sample in samples:
            try:
                outputs = sample['outputs']
                
                # 3D 포인트 클라우드 추출
                pred_points = self.extract_point_cloud(outputs)
                if pred_points is None:
                    continue
                
                # 참조 포인트클라우드 (더미 - 실제로는 GT 데이터 사용)
                ref_points = self.generate_reference_points(sample['prompt'])
                
                if pred_points is not None and ref_points is not None:
                    # PyTorch3D 포인트클라우드 생성
                    pred_pc = Pointclouds(points=[pred_points])
                    ref_pc = Pointclouds(points=[ref_points])
                    
                    # Chamfer Distance 계산
                    cd, _ = chamfer_distance(pred_pc, ref_pc)
                    cd_scores.append(cd.item())
                
            except Exception as e:
                print(f"⚠️ Chamfer Distance 계산 오류: {e}")
                continue
        
        avg_cd = np.mean(cd_scores) if cd_scores else 0.0
        print(f"📊 Chamfer Distance: {avg_cd:.4f}")
        return avg_cd
    
    def calculate_image_metrics(self, samples: List[Dict]) -> Tuple[float, float]:
        """FID/KID 계산"""
        if not TORCHMETRICS_AVAILABLE:
            return 0.0, 0.0
        
        print("🔍 FID/KID 계산 중...")
        
        try:
            # 생성된 이미지들
            generated_images = []
            for sample in samples:
                rendered = self.render_3d_to_2d(sample['outputs'])
                if rendered:
                    generated_images.extend(rendered)
            
            if len(generated_images) < 10:
                print("⚠️ FID/KID 계산에 충분한 이미지가 없습니다.")
                return 0.0, 0.0
            
            # 실제 이미지 (더미 - 실제로는 실제 데이터셋 사용)
            real_images = self.generate_reference_images(len(generated_images))
            
            # 텐서로 변환
            gen_tensor = torch.stack([self.image_to_tensor(img) for img in generated_images])
            real_tensor = torch.stack([self.image_to_tensor(img) for img in real_images])
            
            # FID 계산
            self.fid.update(real_tensor, real=True)
            self.fid.update(gen_tensor, real=False)
            fid_score = self.fid.compute().item()
            
            # KID 계산
            self.kid.update(real_tensor, real=True)
            self.kid.update(gen_tensor, real=False)
            kid_score = self.kid.compute()[0].item()
            
            print(f"📊 FID: {fid_score:.4f}, KID: {kid_score:.4f}")
            return fid_score, kid_score
            
        except Exception as e:
            print(f"⚠️ FID/KID 계산 오류: {e}")
            return 0.0, 0.0
    
    def measure_inference_time(self, model, test_prompts: List[str]) -> float:
        """추론 시간 측정"""
        print("⏱️ 추론 시간 측정 중...")
        
        times = []
        for prompt in test_prompts:
            try:
                torch.cuda.synchronize()
                start_time = time.time()
                
                _ = model.run(prompt, seed=42)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                times.append(end_time - start_time)
                
            except Exception as e:
                print(f"⚠️ 추론 시간 측정 오류: {e}")
                continue
        
        avg_time = np.mean(times) if times else 0.0
        print(f"📊 평균 추론 시간: {avg_time:.2f}초")
        return avg_time
    
    def render_3d_to_2d(self, outputs: Dict) -> Optional[List]:
        """3D 출력을 2D 이미지로 렌더링 (더미 구현)"""
        # 실제로는 TRELLIS의 렌더링 유틸리티 사용
        # 여기서는 더미 이미지 반환
        try:
            from PIL import Image
            dummy_image = Image.new('RGB', (224, 224), color='white')
            return [dummy_image] * 4  # 4개 뷰
        except:
            return None
    
    def extract_point_cloud(self, outputs: Dict) -> Optional[torch.Tensor]:
        """3D 출력에서 포인트 클라우드 추출 (더미 구현)"""
        # 실제로는 TRELLIS의 메시/가우시안에서 포인트 추출
        try:
            # 더미 포인트 클라우드 (1000개 점)
            points = torch.randn(1000, 3).to(self.device)
            return points
        except:
            return None
    
    def generate_reference_points(self, prompt: str) -> Optional[torch.Tensor]:
        """참조 포인트 클라우드 생성 (더미 구현)"""
        # 실제로는 GT 데이터 또는 다른 모델의 출력 사용
        try:
            # 더미 참조 포인트
            points = torch.randn(1000, 3).to(self.device)
            return points
        except:
            return None
    
    def generate_reference_images(self, count: int) -> List:
        """참조 이미지 생성 (더미 구현)"""
        from PIL import Image
        images = []
        for _ in range(count):
            img = Image.new('RGB', (224, 224), color='gray')
            images.append(img)
        return images
    
    def image_to_tensor(self, image) -> torch.Tensor:
        """PIL 이미지를 텐서로 변환"""
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((299, 299)),  # Inception 크기
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).to(self.device)


def evaluate_model(pipeline, config) -> Dict[str, float]:
    """모델 평가 실행"""
    
    # 테스트 프롬프트
    test_prompts = [
        "a red sports car",
        "a wooden chair",
        "a blue coffee mug",
        "a small house",
        "a cute cat",
        "a modern desk lamp",
        "a vintage bicycle",
        "a glass vase with flowers"
    ]
    
    evaluator = TRELLISEvaluator()
    results = evaluator.evaluate_model(pipeline, test_prompts, config)
    
    return results


def calculate_model_efficiency(model: nn.Module) -> Dict[str, Any]:
    """모델 효율성 메트릭 계산"""
    from utils.model_utils import get_trainable_parameters, get_model_memory_usage
    
    trainable_params, total_params = get_trainable_parameters(model)
    memory_info = get_model_memory_usage(model)
    
    efficiency = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'memory_mb': memory_info['total_mb'],
        'parameters_per_mb': total_params / memory_info['total_mb'] if memory_info['total_mb'] > 0 else 0
    }
    
    return efficiency


def compare_models(original_results: Dict, qlora_results: Dict) -> Dict[str, Any]:
    """원본 모델과 QLoRA 모델 비교"""
    
    comparison = {
        'metrics_comparison': {},
        'efficiency_gain': {},
        'quality_retention': {}
    }
    
    # 메트릭 비교
    for metric in ['clip_score', 'chamfer_distance', 'fid_score', 'inference_time']:
        if metric in original_results and metric in qlora_results:
            original_val = original_results[metric]
            qlora_val = qlora_results[metric]
            
            # 값이 낮을수록 좋은 메트릭들 (CD, FID, 추론시간)
            if metric in ['chamfer_distance', 'fid_score', 'inference_time']:
                improvement = (original_val - qlora_val) / original_val * 100
            else:  # 높을수록 좋은 메트릭 (CLIP score)
                improvement = (qlora_val - original_val) / original_val * 100
            
            comparison['metrics_comparison'][metric] = {
                'original': original_val,
                'qlora': qlora_val,
                'improvement_percent': improvement
            }
    
    return comparison


def generate_evaluation_report(results: Dict[str, float], config, save_path: Path):
    """평가 리포트 생성"""
    
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("TRELLIS QLoRA 평가 리포트")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # 모델 정보
    report_lines.append("🔧 모델 정보:")
    report_lines.append(f"  모델: {config.model_name}")
    report_lines.append(f"  LoRA rank: {config.lora_rank}")
    report_lines.append(f"  LoRA alpha: {config.lora_alpha}")
    report_lines.append(f"  훈련 스텝: {config.max_steps}")
    report_lines.append("")
    
    # 평가 결과
    report_lines.append("📊 평가 결과:")
    for metric, value in results.items():
        if metric == 'clip_score':
            report_lines.append(f"  CLIP Score: {value:.4f} (↑ 높을수록 좋음)")
        elif metric == 'chamfer_distance':
            report_lines.append(f"  Chamfer Distance: {value:.4f} (↓ 낮을수록 좋음)")
        elif metric == 'fid_score':
            report_lines.append(f"  FID Score: {value:.4f} (↓ 낮을수록 좋음)")
        elif metric == 'kid_score':
            report_lines.append(f"  KID Score: {value:.4f} (↓ 낮을수록 좋음)")
        elif metric == 'inference_time':
            report_lines.append(f"  추론 시간: {value:.2f}초 (↓ 낮을수록 좋음)")
        else:
            report_lines.append(f"  {metric}: {value:.4f}")
    report_lines.append("")
    
    # 품질 평가
    report_lines.append("🎯 품질 평가:")
    if 'clip_score' in results:
        clip_score = results['clip_score']
        if clip_score >= 0.8:
            report_lines.append("  ✅ 텍스트-3D 일치도: 우수")
        elif clip_score >= 0.6:
            report_lines.append("  🟡 텍스트-3D 일치도: 양호")
        else:
            report_lines.append("  ❌ 텍스트-3D 일치도: 개선 필요")
    
    if 'fid_score' in results:
        fid_score = results['fid_score']
        if fid_score <= 50:
            report_lines.append("  ✅ 렌더링 품질: 우수")
        elif fid_score <= 100:
            report_lines.append("  🟡 렌더링 품질: 양호")
        else:
            report_lines.append("  ❌ 렌더링 품질: 개선 필요")
    
    report_lines.append("")
    report_lines.append("=" * 50)
    
    # 파일 저장
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 평가 리포트 저장: {save_path}")


class QualityMetrics:
    """품질 메트릭 계산 클래스"""
    
    @staticmethod
    def calculate_mesh_quality(mesh_vertices: torch.Tensor, mesh_faces: torch.Tensor) -> Dict[str, float]:
        """메시 품질 메트릭"""
        if not PYTORCH3D_AVAILABLE:
            return {}
        
        try:
            # 기본 메시 통계
            num_vertices = mesh_vertices.shape[0]
            num_faces = mesh_faces.shape[0]
            
            # 엣지 길이 분포
            edges = []
            for face in mesh_faces:
                for i in range(3):
                    v1, v2 = face[i], face[(i+1)%3]
                    edge_length = torch.norm(mesh_vertices[v1] - mesh_vertices[v2])
                    edges.append(edge_length.item())
            
            edge_stats = {
                'mean_edge_length': np.mean(edges),
                'std_edge_length': np.std(edges),
                'min_edge_length': np.min(edges),
                'max_edge_length': np.max(edges)
            }
            
            return {
                'num_vertices': num_vertices,
                'num_faces': num_faces,
                **edge_stats
            }
            
        except Exception as e:
            print(f"⚠️ 메시 품질 계산 오류: {e}")
            return {}
    
    @staticmethod
    def calculate_point_cloud_quality(points: torch.Tensor) -> Dict[str, float]:
        """포인트 클라우드 품질 메트릭"""
        try:
            num_points = points.shape[0]
            
            # 중심점
            centroid = torch.mean(points, dim=0)
            
            # 경계 상자
            bbox_min = torch.min(points, dim=0)[0]
            bbox_max = torch.max(points, dim=0)[0]
            bbox_size = bbox_max - bbox_min
            
            # 밀도 (k-nearest neighbors 기반)
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=10).fit(points.cpu().numpy())
            distances, _ = nbrs.kneighbors(points.cpu().numpy())
            avg_density = np.mean(distances[:, 1:])  # 자기 자신 제외
            
            return {
                'num_points': num_points,
                'bbox_volume': torch.prod(bbox_size).item(),
                'avg_point_density': avg_density,
                'centroid_x': centroid[0].item(),
                'centroid_y': centroid[1].item(),
                'centroid_z': centroid[2].item()
            }
            
        except Exception as e:
            print(f"⚠️ 포인트 클라우드 품질 계산 오류: {e}")
            return {}


def run_comprehensive_evaluation(pipeline, config, save_dir: Path) -> Dict[str, Any]:
    """종합적인 모델 평가"""
    
    print("🔍 종합 평가 시작...")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 기본 평가
    basic_results = evaluate_model(pipeline, config)
    
    # 효율성 평가
    if hasattr(pipeline, 'sparse_structure_decoder'):
        model = pipeline.sparse_structure_decoder
    else:
        model = next(iter(pipeline.models.values()))
    
    efficiency_results = calculate_model_efficiency(model)
    
    # 종합 결과
    comprehensive_results = {
        'quality_metrics': basic_results,
        'efficiency_metrics': efficiency_results,
        'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config_summary': {
            'model_name': config.model_name,
            'lora_rank': config.lora_rank,
            'lora_alpha': config.lora_alpha,
            'max_steps': config.max_steps
        }
    }
    
    # 결과 저장
    import json
    results_file = save_dir / "comprehensive_evaluation.json"
    with open(results_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # 리포트 생성
    report_file = save_dir / "evaluation_report.txt"
    generate_evaluation_report(basic_results, config, report_file)
    
    print("✅ 종합 평가 완료!")
    print(f"📊 결과 저장: {save_dir}")
    
    return comprehensive_results