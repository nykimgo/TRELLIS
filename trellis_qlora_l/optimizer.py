"""
모델 유틸리티 함수들

주요 기능:
- 모델 파라미터 분석
- LoRA 모듈 관리
- 모델 정보 출력
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Any
from peft import PeftModel, LoraConfig


def get_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """훈련 가능한 파라미터 수 계산"""
    trainable_params = 0
    total_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    return trainable_params, total_params


def print_model_info(model: nn.Module, model_name: str = "Model"):
    """모델 정보 출력"""
    trainable_params, total_params = get_trainable_parameters(model)
    trainable_percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    
    print(f"\n📊 {model_name} 정보:")
    print(f"  전체 파라미터: {total_params:,}")
    print(f"  훈련 파라미터: {trainable_params:,}")
    print(f"  훈련 비율: {trainable_percentage:.2f}%")
    
    # 메모리 사용량 추정
    if total_params > 0:
        # FP32 기준 메모리 (바이트)
        memory_fp32 = total_params * 4 / (1024**2)  # MB
        memory_fp16 = total_params * 2 / (1024**2)  # MB
        print(f"  메모리 (FP32): {memory_fp32:.1f} MB")
        print(f"  메모리 (FP16): {memory_fp16:.1f} MB")


def analyze_lora_modules(model: nn.Module) -> Dict[str, Any]:
    """LoRA 모듈 분석"""
    lora_info = {
        "lora_modules": [],
        "base_modules": [],
        "total_lora_params": 0,
        "total_base_params": 0
    }
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
            # LoRA 모듈
            lora_params = sum(p.numel() for p in module.parameters() if 'lora' in str(p))
            lora_info["lora_modules"].append({
                "name": name,
                "type": type(module).__name__,
                "params": lora_params
            })
            lora_info["total_lora_params"] += lora_params
        
        elif len(list(module.parameters())) > 0:
            # 베이스 모듈
            base_params = sum(p.numel() for p in module.parameters())
            lora_info["base_modules"].append({
                "name": name,
                "type": type(module).__name__,
                "params": base_params
            })
            lora_info["total_base_params"] += base_params
    
    return lora_info


def print_lora_info(model: nn.Module):
    """LoRA 정보 출력"""
    lora_info = analyze_lora_modules(model)
    
    print(f"\n🔧 LoRA 모듈 분석:")
    print(f"  LoRA 모듈 수: {len(lora_info['lora_modules'])}")
    print(f"  LoRA 파라미터: {lora_info['total_lora_params']:,}")
    print(f"  베이스 파라미터: {lora_info['total_base_params']:,}")
    
    if lora_info['total_lora_params'] > 0:
        efficiency = lora_info['total_lora_params'] / (lora_info['total_lora_params'] + lora_info['total_base_params']) * 100
        print(f"  LoRA 효율성: {efficiency:.2f}%")
    
    # 상위 LoRA 모듈들 출력
    if lora_info['lora_modules']:
        print(f"\n  주요 LoRA 모듈:")
        sorted_modules = sorted(lora_info['lora_modules'], key=lambda x: x['params'], reverse=True)
        for module in sorted_modules[:5]:  # 상위 5개만
            print(f"    - {module['name']}: {module['params']:,} 파라미터")


def freeze_base_model(model: nn.Module):
    """베이스 모델 파라미터 동결"""
    for name, param in model.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False
    
    print("🔒 베이스 모델 파라미터 동결 완료")


def unfreeze_lora_parameters(model: nn.Module):
    """LoRA 파라미터만 해동"""
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    
    print("🔓 LoRA 파라미터 해동 완료")


def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
    """모델 메모리 사용량 계산"""
    param_memory = 0
    buffer_memory = 0
    
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_memory += buffer.numel() * buffer.element_size()
    
    total_memory = param_memory + buffer_memory
    
    return {
        "parameters_mb": param_memory / (1024**2),
        "buffers_mb": buffer_memory / (1024**2),
        "total_mb": total_memory / (1024**2)
    }


def compare_models(original_model: nn.Module, lora_model: nn.Module):
    """원본 모델과 LoRA 모델 비교"""
    print("\n📊 모델 비교:")
    
    # 파라미터 수 비교
    orig_trainable, orig_total = get_trainable_parameters(original_model)
    lora_trainable, lora_total = get_trainable_parameters(lora_model)
    
    print(f"  원본 모델:")
    print(f"    - 전체: {orig_total:,}")
    print(f"    - 훈련: {orig_trainable:,}")
    
    print(f"  LoRA 모델:")
    print(f"    - 전체: {lora_total:,}")
    print(f"    - 훈련: {lora_trainable:,}")
    
    # 메모리 사용량 비교
    orig_memory = get_model_memory_usage(original_model)
    lora_memory = get_model_memory_usage(lora_model)
    
    print(f"  메모리 사용량:")
    print(f"    - 원본: {orig_memory['total_mb']:.1f} MB")
    print(f"    - LoRA: {lora_memory['total_mb']:.1f} MB")
    
    # 효율성 계산
    param_reduction = (1 - lora_trainable / orig_trainable) * 100 if orig_trainable > 0 else 0
    memory_reduction = (1 - lora_memory['total_mb'] / orig_memory['total_mb']) * 100 if orig_memory['total_mb'] > 0 else 0
    
    print(f"  효율성:")
    print(f"    - 훈련 파라미터 감소: {param_reduction:.1f}%")
    print(f"    - 메모리 사용량 차이: {memory_reduction:.1f}%")


def find_target_modules(model: nn.Module, module_types: List[str] = None) -> List[str]:
    """LoRA 적용 가능한 타겟 모듈 찾기"""
    if module_types is None:
        module_types = ["Linear", "Conv1d", "Conv2d", "Conv3d"]
    
    target_modules = []
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in module_types:
            # 일반적인 transformer 패턴 확인
            if any(pattern in name.lower() for pattern in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
                target_modules.append(name.split('.')[-1])  # 마지막 부분만
    
    # 중복 제거
    target_modules = list(set(target_modules))
    
    print(f"🎯 발견된 타겟 모듈: {target_modules}")
    return target_modules


def calculate_lora_parameters(rank: int, input_dim: int, output_dim: int) -> int:
    """LoRA 파라미터 수 계산"""
    # LoRA: W = W_0 + BA (B: output_dim x rank, A: rank x input_dim)
    lora_params = rank * (input_dim + output_dim)
    return lora_params


def estimate_lora_memory(model: nn.Module, lora_config: LoraConfig) -> Dict[str, float]:
    """LoRA 메모리 사용량 추정"""
    total_lora_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target in name for target in lora_config.target_modules):
            in_features = module.in_features
            out_features = module.out_features
            lora_params = calculate_lora_parameters(lora_config.r, in_features, out_features)
            total_lora_params += lora_params
    
    # 메모리 계산 (FP16 기준)
    lora_memory_mb = total_lora_params * 2 / (1024**2)
    
    return {
        "lora_parameters": total_lora_params,
        "lora_memory_mb": lora_memory_mb
    }


def validate_lora_config(model: nn.Module, lora_config: LoraConfig) -> bool:
    """LoRA 설정 검증"""
    print("🔍 LoRA 설정 검증...")
    
    # 타겟 모듈 존재 확인
    found_modules = []
    for name, module in model.named_modules():
        if any(target in name for target in lora_config.target_modules):
            found_modules.append(name)
    
    if not found_modules:
        print(f"❌ 타겟 모듈을 찾을 수 없습니다: {lora_config.target_modules}")
        return False
    
    print(f"✅ 발견된 타겟 모듈: {len(found_modules)}개")
    
    # 랭크 검증
    if lora_config.r <= 0:
        print(f"❌ 잘못된 LoRA rank: {lora_config.r}")
        return False
    
    # 알파 검증
    if lora_config.lora_alpha <= 0:
        print(f"❌ 잘못된 LoRA alpha: {lora_config.lora_alpha}")
        return False
    
    print("✅ LoRA 설정 검증 완료")
    return True


def save_model_info(model: nn.Module, save_path: str):
    """모델 정보를 파일로 저장"""
    import json
    from pathlib import Path
    
    trainable_params, total_params = get_trainable_parameters(model)
    memory_info = get_model_memory_usage(model)
    lora_info = analyze_lora_modules(model)
    
    model_info = {
        "parameters": {
            "total": total_params,
            "trainable": trainable_params,
            "trainable_percentage": (trainable_params / total_params * 100) if total_params > 0 else 0
        },
        "memory": memory_info,
        "lora": {
            "modules_count": len(lora_info['lora_modules']),
            "lora_parameters": lora_info['total_lora_params'],
            "base_parameters": lora_info['total_base_params']
        },
        "architecture": str(model)
    }
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    print(f"💾 모델 정보 저장: {save_path}")


class ModelProfiler:
    """모델 프로파일링 클래스"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
    
    def register_hooks(self):
        """훅 등록"""
        def forward_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = {
                        'shape': output.shape,
                        'memory_mb': output.numel() * output.element_size() / (1024**2)
                    }
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = {
                        'shape': grad_output[0].shape,
                        'memory_mb': grad_output[0].numel() * grad_output[0].element_size() / (1024**2)
                    }
            return hook
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # leaf modules only
                self.hooks.append(module.register_forward_hook(forward_hook(name)))
                self.hooks.append(module.register_backward_hook(backward_hook(name)))
    
    def clear_hooks(self):
        """훅 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """프로파일링 요약"""
        total_activation_memory = sum(info['memory_mb'] for info in self.activations.values())
        total_gradient_memory = sum(info['memory_mb'] for info in self.gradients.values())
        
        return {
            'activation_memory_mb': total_activation_memory,
            'gradient_memory_mb': total_gradient_memory,
            'total_memory_mb': total_activation_memory + total_gradient_memory,
            'module_count': len(self.activations)
        }
    
    def __enter__(self):
        self.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear_hooks()