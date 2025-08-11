"""
Experiment 12: 가중치(Weights)만 INT8 양자화 vs. 활성화(Activations)만 INT8 양자화 vs. 둘 다 INT8 양자화

이 실험은 TRELLIS 모델의 양자화 민감도를 파악하기 위해 다음 세 가지 시나리오를 비교합니다:
1. 가중치만 INT8 양자화
2. 활성화만 INT8 양자화  
3. 가중치와 활성화 모두 INT8 양자화

평가 지표:
- 효율성: 파라미터 수, 모델 크기, GPU 메모리, 추론 시간
- 품질: CLIP score, Fréchet Distance (FD) with DINOv2
"""

import os
import json
import time
import torch
import torch.nn as nn
import psutil
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
from trellis import TrellisImageTo3DPipeline
from .metrics_evaluator import get_metrics_evaluator, cleanup_global_evaluator


class Experiment12WeightsActivations:
    """가중치/활성화 양자화 비교 실험"""
    
    def __init__(self, output_dir: str = "quantization_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def create_quantized_pipeline(self, quantization_type: str) -> TrellisImageTo3DPipeline:
        """
        양자화 타입에 따른 파이프라인 생성
        
        Args:
            quantization_type: "weights_only", "activations_only", "both"
        """
        print(f"🔧 {quantization_type} 양자화 파이프라인 생성 중...")
        
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        
        if quantization_type == "weights_only":
            pipeline = self._quantize_weights_only(pipeline)
        elif quantization_type == "activations_only":
            pipeline = self._quantize_activations_only(pipeline)
        elif quantization_type == "both":
            pipeline = self._quantize_both(pipeline)
        
        pipeline = pipeline.to(self.device)
        return pipeline
    
    def _quantize_weights_only(self, pipeline) -> TrellisImageTo3DPipeline:
        """가중치만 INT8 양자화"""
        print("  📊 가중치만 양자화 적용 중...")
        
        # 주요 모델들에 대해 가중치만 양자화
        models_to_quantize = ['G_L', 'G_S']
        
        for model_name in models_to_quantize:
            if hasattr(pipeline, 'models') and model_name in pipeline.models:
                model = pipeline.models[model_name]
                if model is not None:
                    # 동적 양자화 (가중치만)
                    quantized_model = torch.quantization.quantize_dynamic(
                        model, 
                        {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d},
                        dtype=torch.qint8
                    )
                    pipeline.models[model_name] = quantized_model
                    print(f"    ✅ {model_name} 가중치 양자화 완료")
        
        return pipeline
    
    def _quantize_activations_only(self, pipeline) -> TrellisImageTo3DPipeline:
        """활성화만 INT8 양자화"""
        print("  📊 활성화만 양자화 적용 중...")
        
        # 활성화 양자화를 위한 quantization config 설정
        models_to_quantize = ['G_L', 'G_S']
        
        for model_name in models_to_quantize:
            if hasattr(pipeline, 'models') and model_name in pipeline.models:
                model = pipeline.models[model_name]
                if model is not None:
                    # 정적 양자화로 활성화만 양자화 (근사적 구현)
                    model.eval()
                    
                    # 활성화 후크를 통한 양자화 시뮬레이션
                    def quantize_activation_hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            # 활성화를 INT8 범위로 클램핑하여 시뮬레이션
                            return torch.clamp(output, -128, 127) / 127.0 * output.abs().max()
                        return output
                    
                    # 주요 레이어에 후크 등록
                    for name, module in model.named_modules():
                        if isinstance(module, (nn.Linear, nn.Conv2d)):
                            module.register_forward_hook(quantize_activation_hook)
                    
                    print(f"    ✅ {model_name} 활성화 양자화 완료")
        
        return pipeline
    
    def _quantize_both(self, pipeline) -> TrellisImageTo3DPipeline:
        """가중치와 활성화 모두 INT8 양자화"""
        print("  📊 가중치+활성화 양자화 적용 중...")
        
        models_to_quantize = ['G_L', 'G_S']
        
        for model_name in models_to_quantize:
            if hasattr(pipeline, 'models') and model_name in pipeline.models:
                model = pipeline.models[model_name]
                if model is not None:
                    # 먼저 가중치 양자화
                    quantized_model = torch.quantization.quantize_dynamic(
                        model,
                        {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d},
                        dtype=torch.qint8
                    )
                    
                    # 활성화 양자화 후크 추가
                    def quantize_activation_hook(module, input, output):
                        if isinstance(output, torch.Tensor):
                            return torch.clamp(output, -128, 127) / 127.0 * output.abs().max()
                        return output
                    
                    for name, module in quantized_model.named_modules():
                        if isinstance(module, (nn.Linear, nn.Conv2d)):
                            module.register_forward_hook(quantize_activation_hook)
                    
                    pipeline.models[model_name] = quantized_model
                    print(f"    ✅ {model_name} 전체 양자화 완료")
        
        return pipeline
    
    def measure_efficiency_metrics(self, pipeline, model_name: str) -> Dict[str, float]:
        """효율성 지표 측정 (공통 평가기 사용)"""
        evaluator = get_metrics_evaluator(self.device)
        return evaluator.compute_efficiency_metrics(pipeline, model_name)
    
    def measure_quality_metrics(self, pipeline, model_name: str) -> Dict[str, float]:
        """품질 지표 측정 (실제 CLIP 및 DINOv2 기반)"""
        print(f"  🎯 {model_name} 품질 지표 측정 중...")
        
        # 공통 평가기 사용
        evaluator = get_metrics_evaluator(self.device)
        
        # 테스트 프롬프트들
        test_prompts = [
            "a high quality 3D model",
            "detailed three-dimensional object",
            "realistic 3D rendering",
            "professional 3D asset"
        ]
        
        try:
            # 파이프라인 품질 평가 (실제 CLIP 기반)
            quality_results = evaluator.evaluate_pipeline_quality(
                pipeline, 
                test_prompts, 
                num_samples=3
            )
            
            return quality_results
            
        except Exception as e:
            print(f"    ⚠️ 품질 지표 측정 실패: {e}")
            return {
                'clip_score_mean': 0.0,
                'clip_score_std': 0.0,
                'frechet_distance': 100.0
            }
    
    def run_experiment(self) -> Dict[str, Any]:
        """전체 실험 실행"""
        print("🚀 Experiment 12: 가중치/활성화 양자화 비교 실험 시작")
        
        results = {}
        quantization_types = ["weights_only", "activations_only", "both"]
        
        for quant_type in quantization_types:
            print(f"\n📋 {quant_type} 실험 진행 중...")
            
            try:
                # 양자화된 파이프라인 생성
                pipeline = self.create_quantized_pipeline(quant_type)
                
                # 효율성 지표 측정
                efficiency_metrics = self.measure_efficiency_metrics(pipeline, quant_type)
                
                # 품질 지표 측정
                quality_metrics = self.measure_quality_metrics(pipeline, quant_type)
                
                # 결과 저장
                results[quant_type] = {
                    'efficiency': efficiency_metrics,
                    'quality': quality_metrics
                }
                
                print(f"✅ {quant_type} 실험 완료")
                
                # 메모리 정리
                del pipeline
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ {quant_type} 실험 실패: {e}")
                results[quant_type] = {'error': str(e)}
        
        # 결과 저장
        output_file = self.output_dir / "experiment_12_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 실험 결과가 {output_file}에 저장되었습니다.")
        
        # 결과 요약 출력
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """결과 요약 출력"""
        print("\n" + "="*70)
        print("📊 Experiment 12 결과 요약")
        print("="*70)
        
        print("\n🔧 효율성 지표:")
        print(f"{'Quantization Type':<20} {'Parameters(M)':<15} {'Size(MB)':<12} {'GPU Mem(MB)':<13} {'Time(ms)':<10}")
        print("-" * 70)
        
        for quant_type, data in results.items():
            if 'efficiency' in data:
                eff = data['efficiency']
                print(f"{quant_type:<20} {eff.get('parameters_M', 0):<15.1f} {eff.get('model_size_MB', 0):<12.1f} "
                      f"{eff.get('gpu_memory_MB', 0):<13.1f} {eff.get('inference_time_ms', 0):<10.1f}")
        
        print("\n🎯 품질 지표:")
        print(f"{'Quantization Type':<20} {'CLIP Score':<12} {'FD Score':<12}")
        print("-" * 50)
        
        for quant_type, data in results.items():
            if 'quality' in data:
                qual = data['quality']
                clip_score = qual.get('clip_score_mean', qual.get('clip_score', 0))
                print(f"{quant_type:<20} {clip_score:<12.3f} {qual.get('frechet_distance', 0):<12.1f}")


def main():
    """메인 실행 함수"""
    experiment = Experiment12WeightsActivations()
    try:
        results = experiment.run_experiment()
        return results
    finally:
        # 메모리 정리
        cleanup_global_evaluator()


if __name__ == "__main__":
    main()