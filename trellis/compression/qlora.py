import argparse
from typing import Iterable, Optional

import torch
import torch.nn as nn

try:
    import bitsandbytes as bnb
except ImportError:  # pragma: no cover - dependency may not be installed during linting
    bnb = None

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
except ImportError:  # pragma: no cover - dependency may not be installed during linting
    LoraConfig = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def _quantize_module(module: nn.Module) -> nn.Module:
    """Recursively quantize Linear layers to 4-bit using bitsandbytes."""
    if bnb is None:
        raise ImportError("bitsandbytes is required for QLoRA quantization")

    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            qlinear = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
            )
            qlinear.weight = child.weight
            if child.bias is not None:
                qlinear.bias = child.bias
            setattr(module, name, qlinear)
        else:
            _quantize_module(child)
    return module


def apply_qlora(
    model: nn.Module,
    *,
    target_modules: Optional[Iterable[str]] = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> nn.Module:
    """Attach LoRA adapters to a quantized model."""
    if any(x is None for x in (LoraConfig, get_peft_model, prepare_model_for_kbit_training)):
        raise ImportError("peft is required for QLoRA quantization")

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="UNDEFINED",
    )
    return get_peft_model(model, lora_config)


def load_quantized_trellis_pipeline(
    model_name: str = "microsoft/TRELLIS-text-large",
    *,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
):
    """Load a TRELLIS text-to-3D pipeline with QLoRA quantization."""
    from trellis.pipelines import TrellisTextTo3DPipeline

    pipeline = TrellisTextTo3DPipeline.from_pretrained(model_name)
    for key, module in pipeline.models.items():
        module = _quantize_module(module)
        pipeline.models[key] = apply_qlora(
            module,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
    pipeline.cuda()
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Apply QLoRA quantization to TRELLIS models")
    parser.add_argument("--model", default="microsoft/TRELLIS-text-large", help="Model name or path")
    parser.add_argument("--output", required=True, help="Directory to save LoRA adapters")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--dropout", type=float, default=0.0, help="LoRA dropout")
    args = parser.parse_args()

    pipeline = load_quantized_trellis_pipeline(
        args.model,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
    )

    for name, model in pipeline.models.items():
        model.save_pretrained(f"{args.output}/{name}")


if __name__ == "__main__":
    main()
