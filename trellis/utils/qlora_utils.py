import torch
import torch.nn as nn
from typing import List
from contextlib import contextmanager
import types
import math

try:
    import bitsandbytes as bnb  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bnb = None

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from ..modules import sparse as sp
from ..modules.sparse.linear import SparseLinear

from ..utils.elastic_utils import ElasticModuleMixin  # Mixin 클래스 import

import torch.nn.functional as F

class SparseDropout(nn.Module):
    """SparseTensor와 호환되는 드롭아웃 레이어."""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        # 학습 중이 아니거나 드롭아웃 확률이 0이면 아무것도 하지 않음
        if not self.training or self.p == 0:
            return input
        
        # 입력이 SparseTensor인 경우, 그 내용물(feats)에만 드롭아웃 적용
        if isinstance(input, sp.SparseTensor):
            new_feats = F.dropout(input.feats, p=self.p, training=self.training)
            return input.replace(feats=new_feats)
        
        # 일반 텐서는 그대로 드롭아웃 적용
        return F.dropout(input, p=self.p, training=self.training)

def _replace_lora_dropout_modules(module: nn.Module):
    """
    LoraLayer를 순회하며 내부의 Dropout 모듈을 SparseDropout으로 교체합니다.
    """
    from peft.tuners.lora import LoraLayer
    
    for child_module in module.children():
        # LoraLayer를 찾음
        if isinstance(child_module, LoraLayer):
            # lora_dropout 속성과 ModuleDict가 있는지 확인
            if hasattr(child_module, 'lora_dropout') and isinstance(child_module.lora_dropout, nn.ModuleDict):
                for adapter_name, dropout_layer in child_module.lora_dropout.items():
                    # nn.Dropout 인스턴스를 찾아서
                    if isinstance(dropout_layer, nn.Dropout):
                        p = dropout_layer.p
                        # 우리가 만든 SparseDropout으로 교체
                        child_module.lora_dropout[adapter_name] = SparseDropout(p)
        else:
            # 다른 모든 모듈에 대해 재귀적으로 함수 호출
            _replace_lora_dropout_modules(child_module)

def _new_lora_linear_forward(self, x):
    """
    SparseTensor를 인식하는 새로운 nn.Linear.forward 메서드.
    'self'는 nn.Linear의 인스턴스입니다.
    """
    if isinstance(x, sp.SparseTensor):
        # 입력이 SparseTensor이면, 그 내용물(feats)에만 linear 연산을 적용
        return x.replace(feats=F.linear(x.feats, self.weight, self.bias))
    # 일반 텐서는 원래의 linear 연산을 그대로 수행
    return F.linear(x, self.weight, self.bias)

def _patch_lora_linear_layers(module: nn.Module):
    """
    모델을 순회하며 모든 LoraLayer의 lora_A, lora_B 선형 계층의
    forward 메서드를 SparseTensor를 인식하는 버전으로 교체(몽키 패치)합니다.
    """
    from peft.tuners.lora import LoraLayer
    import types
    
    for child_module in module.children():
        # LoraLayer (QuantizedLinear, SparseLinear 등을 감싸는)를 찾음
        if isinstance(child_module, LoraLayer):
            # lora_A 모듈이 있다면, 모든 어댑터에 대해 패치 적용
            if hasattr(child_module, 'lora_A'):
                for adapter in child_module.lora_A.values():
                    adapter.forward = types.MethodType(_new_lora_linear_forward, adapter)
            # lora_B 모듈이 있다면, 모든 어댑터에 대해 패치 적용
            if hasattr(child_module, 'lora_B'):
                for adapter in child_module.lora_B.values():
                    adapter.forward = types.MethodType(_new_lora_linear_forward, adapter)
        else:
            # 다른 모든 모듈에 대해 재귀적으로 함수 호출
            _patch_lora_linear_layers(child_module)




class SparseLinear4bit(bnb.nn.Linear4bit):  # type: ignore
    """4bit quantized linear layer that accepts a :class:`SparseTensor`."""

    def forward(self, input: sp.SparseTensor):  # type: ignore[override]
        return input.replace(super().forward(input.feats))


def _replace_linear(module: nn.Module) -> None:
    """Recursively replace Linear/SparseLinear with 4bit versions."""
    if bnb is None:
        return
    for name, child in list(module.named_children()):
        if isinstance(child, SparseLinear):
            new_linear = SparseLinear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
            )
            new_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_linear.bias.data.copy_(child.bias.data)
            setattr(module, name, new_linear)
        elif isinstance(child, nn.Linear):
            new_linear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
            )
            new_linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_linear.bias.data.copy_(child.bias.data)
            setattr(module, name, new_linear)
        else:
            _replace_linear(child)


def _get_target_modules_for_trellis(model_type: str = "structured_latent_flow") -> List[str]:
    """TRELLIS 𝓖_L 모델에 특화된 타겟 모듈 패턴"""
    if model_type == "structured_latent_flow":
        return [
            # Self/Cross attention projections in transformer blocks
            "blocks.*.attn.q_proj",
            "blocks.*.attn.k_proj", 
            "blocks.*.attn.v_proj",
            "blocks.*.attn.o_proj",
            # FFN layers in transformer blocks
            "blocks.*.mlp.mlp.0",  # First linear layer in FFN
            "blocks.*.mlp.mlp.2",  # Second linear layer in FFN
            # Input/Output blocks attention and FFN
            "input_blocks.*.attn.*_proj",
            "input_blocks.*.mlp.*",
            "out_blocks.*.attn.*_proj", 
            "out_blocks.*.mlp.*",
            # Main input/output projections
            "input_layer",
            "out_layer",
            # ResBlock skip connections (SparseLinear)
            "*.skip_connection",
        ]
    else:
        # 다른 모델 타입을 위한 일반적인 패턴
        return ["*.proj", "*.linear", "*.mlp.*"]


def _find_linear_module_names(model: nn.Module, use_patterns: bool = True, model_type: str = "structured_latent_flow") -> List[str]:
    """Finds the names of all linear and sparse linear modules in the model."""
    if use_patterns:
        # TRELLIS에 특화된 패턴 사용
        target_patterns = _get_target_modules_for_trellis(model_type)
        names = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, SparseLinear)):
                # 패턴 매칭 확인
                for pattern in target_patterns:
                    import fnmatch
                    if fnmatch.fnmatch(name, pattern):
                        names.append(name)
                        break
        return names
    else:
        # 기존 방식: 모든 Linear 레이어
        names = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, SparseLinear)):
                names.append(name)
        return names


class ElasticPeftModel(ElasticModuleMixin):
    """
    PeftModel의 ElasticModuleMixin 호환 래퍼
    저수준 Elastic 상호작용을 완전히 지원
    DDP/Peft/Sampler의 복잡한 인자 전달 문제를 해결하는 '인자 방화벽' 역할을 수행합니다.
    """
    
    def __init__(self, peft_model, original_model):
        super().__init__()  # <--- 상속받은 Mixin의 __init__ 메서드 호출
        # PeftModel의 모든 속성과 메서드를 위임
        self.__dict__['_peft_model'] = peft_model
        self.__dict__['_original_model'] = original_model
        '''
        # DDP 호환성을 위한 설정
        self._ddp_params_and_buffers_to_ignore = set()
        
        # Frozen 파라미터들을 DDP에서 무시하도록 설정
        for name, param in peft_model.named_parameters():
            if not param.requires_grad:
                self._ddp_params_and_buffers_to_ignore.add(name)
        '''
        
    def __getattr__(self, name):
        """PeftModel의 속성/메서드에 접근"""
        return getattr(self._peft_model, name)
    
    def __setattr__(self, name, value):
        """속성 설정을 PeftModel에 위임"""
        if name.startswith('_') or name in ['_peft_model', '_original_model']:
            self.__dict__[name] = value
        else:
            setattr(self._peft_model, name, value)
    
    def _get_input_size(self, x, *args, **kwargs):
        """입력 크기 계산"""
        if hasattr(x, 'feats'):  # SparseTensor
            return x.feats.shape[0]
        return x.shape[0]

    @contextmanager
    def with_mem_ratio(self, mem_ratio=1.0):
        """
        메모리 비율에 따른 gradient checkpointing 설정
        원본 모델의 blocks에 직접 접근하여 제어
        """
        if mem_ratio == 1.0:
            yield 1.0
            return
        
        # 원본 모델에서 blocks 접근
        base_model = self._peft_model.base_model.model  # PeftModel -> base_model -> 실제 모델
        
        if not hasattr(base_model, 'blocks'):
            # blocks가 없으면 원본 모델의 with_mem_ratio 사용
            if hasattr(self._original_model, 'with_mem_ratio'):
                with self._original_model.with_mem_ratio(mem_ratio) as exact_ratio:
                    yield exact_ratio
            else:
                yield mem_ratio
            return
        
        # SparseTransformerElasticMixin과 동일한 로직
        num_blocks = len(base_model.blocks)
        num_checkpoint_blocks = min(math.ceil((1 - mem_ratio) * num_blocks) + 1, num_blocks)
        exact_mem_ratio = 1 - (num_checkpoint_blocks - 1) / num_blocks
        
        # 원본 상태 저장
        original_checkpoints = []
        for i in range(num_blocks):
            original_checkpoints.append(base_model.blocks[i].use_checkpoint)
            base_model.blocks[i].use_checkpoint = i < num_checkpoint_blocks
        
        try:
            yield exact_mem_ratio
        finally:
            # 원본 상태 복원
            for i in range(num_blocks):
                base_model.blocks[i].use_checkpoint = original_checkpoints[i]
    
    def register_memory_controller(self, memory_controller):
        """메모리 컨트롤러 등록"""
        self._memory_controller = memory_controller
        # 원본 모델에도 전달 (호환성)
        if hasattr(self._original_model, 'register_memory_controller'):
            self._original_model.register_memory_controller(memory_controller)

    def __call__(self, *args, **kwargs):
        """
        어떤 형태의 인자가 들어오든 forward로 전달합니다.
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        '인자 밀수' 작전의 시작점.
        x, t, cond를 'smuggled_args' 가방에 담아 Peft가 건드리지 못하게 전달합니다.
        """
        # Sampler로부터 받은 인자를 추출
        x = args[0]
        t = args[1]
        cond = args[2]

        # 모든 인자를 하나의 딕셔너리에 포장
        smuggled_args = {'x': x, 't': t, 'cond': cond}

        # --- Elastic 로직 (이전과 동일) ---
        if (self._memory_controller is None or
            not torch.is_grad_enabled() or
            not self.training):
            # **핵심**: 인자들을 'smuggled_args' 키워드 인자 하나로 전달
            return self._peft_model(smuggled_args=smuggled_args)
        else:
            # Elastic forward
            input_size = self._get_input_size(x)
            mem_ratio = self._memory_controller.get_mem_ratio(input_size)
            with self.with_mem_ratio(mem_ratio) as exact_mem_ratio:
                # **핵심**: 인자들을 'smuggled_args' 키워드 인자 하나로 전달
                ret = self._peft_model(smuggled_args=smuggled_args)
            self._memory_controller.update_run_states(input_size, exact_mem_ratio)
            return ret

    def state_dict(self, *args, **kwargs):
        """state_dict를 PeftModel에 위임"""
        return self._peft_model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, *args, **kwargs):
        """load_state_dict를 PeftModel에 위임"""
        return self._peft_model.load_state_dict(*args, **kwargs)
    
    def parameters(self, *args, **kwargs):
        """parameters를 PeftModel에 위임"""
        return self._peft_model.parameters(*args, **kwargs)
    
    def named_parameters(self, *args, **kwargs):
        """named_parameters를 PeftModel에 위임"""
        return self._peft_model.named_parameters(*args, **kwargs)
    
    def train(self, mode=True):
        """train 모드를 PeftModel에 위임"""
        return self._peft_model.train(mode)
    
    def eval(self):
        """eval 모드를 PeftModel에 위임"""
        return self._peft_model.eval()
    
    def to(self, *args, **kwargs):
        """디바이스 이동을 PeftModel에 위임"""
        result = self._peft_model.to(*args, **kwargs)
        return self  # 래퍼 자체를 반환
    
    def cuda(self, *args, **kwargs):
        """CUDA 이동을 PeftModel에 위임"""
        self._peft_model.cuda(*args, **kwargs)
        return self


def apply_qlora(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    quantize: bool = True,
    use_target_patterns: bool = True,
    model_type: str = "structured_latent_flow",
    verbose: bool = True,
) -> nn.Module:
    """Apply QLoRA to a model.

    Args:
        model: Model to modify in-place.
        r: LoRA rank.
        lora_alpha: LoRA alpha scaling.
        lora_dropout: LoRA dropout.
        quantize: If ``True`` and bitsandbytes is available, replace linear
            layers with 4-bit quantized versions before attaching LoRA adapters.
        use_target_patterns: If ``True``, use TRELLIS-specific target patterns.
        model_type: Type of model for pattern selection.
        verbose: If ``True``, print detailed information.
    Returns:
        The modified model.
    """
    
    # 원본 디바이스 저장
    original_device = next(model.parameters()).device
    
    # Elastic 속성 체크
    is_elastic = isinstance(model, ElasticModuleMixin)
    original_model = model  # 원본 참조 저장
    
    if is_elastic and verbose:
        print("Warning: Elastic model detected. Creating full Elastic-compatible wrapper...")
    
    if quantize and bnb is not None:
        _replace_linear(model)
        model = prepare_model_for_kbit_training(model)

    target_modules = _find_linear_module_names(model, use_target_patterns, model_type)
    
    # 디버깅을 위한 타겟 모듈 출력
    if verbose:
        print(f"QLoRA will be applied to {len(target_modules)} modules:")
        for module_name in target_modules[:10]:  # 처음 10개만 출력
            print(f"  - {module_name}")
        if len(target_modules) > 10:
            print(f"  ... and {len(target_modules) - 10} more modules")
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="FEATURE_EXTRACTION",  # 3D 생성 모델에 적합한 task_type
    )
    peft_model = get_peft_model(model, lora_config)
    
    # LoRA 레이어의 Dropout을 Sparse-aware 버전으로 교체
    _replace_lora_dropout_modules(peft_model)
    # LoRA 레이어의 Linear를 Sparse-aware 버전으로 교체 (몽키 패치)
    _patch_lora_linear_layers(peft_model)

    # Elastic 호환성이 필요한 경우 완전한 래퍼 생성
    if is_elastic:
        elastic_model = ElasticPeftModel(peft_model, original_model)
        model = elastic_model
        
        '''
        if verbose:
            print("   ✅ Full Elastic compatibility wrapper created")
            print("   🔧 Supports: blocks manipulation, memory controller, checkpointing")
        '''
    else:
        model = peft_model
    
    # QLoRA 적용 후 모든 파라미터를 원본 디바이스로 이동
    model = model.to(original_device)
    
    # DDP 호환성을 위해 모든 파라미터가 같은 디바이스에 있는지 확인
    devices = {p.device for p in model.parameters()}
    if len(devices) > 1:
        '''if verbose:
            print(f"Warning: Parameters on multiple devices: {devices}")
            print("Moving all parameters to CUDA...")'''
        model = model.cuda()
    
    return model