"""TRELLIS 모델 로더 및 QLoRA 어댑터"""

from typing import Tuple


class ModelLoader:
    """사전학습된 TRELLIS 모델을 로드하고 QLoRA를 적용"""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def load_model(self):
        """모델과 토크나이저 로드"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quant_config,
            device_map="auto",
        )

        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=None,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # pragma: no cover - 콘솔 출력 목적
        return model, tokenizer
