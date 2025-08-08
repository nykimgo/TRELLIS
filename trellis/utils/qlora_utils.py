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
        '인자 방화벽' 역할을 수행합니다.
        DDP/Peft에 의해 인자가 어떻게 망가지든, (x, t, cond) 형태를 완벽히 복원하고
        불필요한 kwargs는 모두 걸러냅니다.
        """
        # ========================= DEBUGGING CODE START =========================
        import torch
        # 여러 GPU에서 로그가 섞이지 않도록 rank 0에서만 출력
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("\n\n" + "="*60)
            print("🕵️  DEBUG: ENTERING ElasticPeftModel.forward")
            print(f"Received {len(args)} positional args (args):")
            for i, arg in enumerate(args):
                print(f"  - args[{i}]: type={type(arg)}", end="")
                if isinstance(arg, torch.Tensor):
                    print(f", shape={arg.shape}, device={arg.device}")
                elif isinstance(arg, dict):
                    print(f", DICT KEYS={list(arg.keys())}")
                else:
                    print()
            print(f"Received {len(kwargs)} keyword args (kwargs): {list(kwargs.keys())}")
            print("="*60 + "\n\n")
        # ========================== DEBUGGING CODE END ==========================
        # Sampler로부터 받은 모든 인자 중, 위치 인자 3개만 사용합니다.
        x = args[0]
        t = args[1]
        
        # `cond`는 딕셔너리가 아닌 텐서일 수 있습니다. 
        # Sampler의 CFG Mixin이 딕셔너리를 풀어서 텐서만 전달하는 경우가 많습니다.
        # 따라서 세 번째 위치 인자를 그대로 `cond`로 사용합니다.
        cond = args[2]

        # --- Elastic 로직 (이전과 동일) ---
        if (self._memory_controller is None or
            not torch.is_grad_enabled() or
            not self.training):
            # **핵심**: PeftModel 호출 시, 오직 (x, t, cond)만 전달하고 kwargs를 전달하지 않음.
            return self._peft_model(x, t, cond)
        else:
            # Elastic forward
            input_size = self._get_input_size(x)
            mem_ratio = self._memory_controller.get_mem_ratio(input_size)
            with self.with_mem_ratio(mem_ratio) as exact_mem_ratio:
                # **핵심**: PeftModel 호출 시, 오직 (x, t, cond)만 전달하고 kwargs를 전달하지 않음.
                ret = self._peft_model(x, t, cond)
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
    from ..utils.elastic_utils import ElasticModuleMixin
    
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
    
    # Elastic 호환성이 필요한 경우 완전한 래퍼 생성
    if is_elastic:
        elastic_model = ElasticPeftModel(peft_model, original_model)
        model = elastic_model
        
        if verbose:
            print("   ✅ Full Elastic compatibility wrapper created")
            print("   🔧 Supports: blocks manipulation, memory controller, checkpointing")
    else:
        model = peft_model
    
    # QLoRA 적용 후 모든 파라미터를 원본 디바이스로 이동
    model = model.to(original_device)
    
    # DDP 호환성을 위해 모든 파라미터가 같은 디바이스에 있는지 확인
    devices = {p.device for p in model.parameters()}
    if len(devices) > 1:
        if verbose:
            print(f"Warning: Parameters on multiple devices: {devices}")
            print("Moving all parameters to CUDA...")
        model = model.cuda()
    
    return model