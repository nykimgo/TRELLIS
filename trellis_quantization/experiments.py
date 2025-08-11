"""TRELLIS 양자화 실험 스크립트"""

from pathlib import Path
from typing import List

from quantization_manager import TRELLISQuantizationManager


# 실험 12: 가중치만, 활성화만, 둘 다 INT8 양자화 비교

def run_experiment_12(model_name: str = "text-base", base_output: str = "quantization_results") -> List[Path]:
    results = []
    configs = [
        ("weights", dict(quantize_weights=True, quantize_activations=False)),
        ("activations", dict(quantize_weights=False, quantize_activations=True)),
        ("both", dict(quantize_weights=True, quantize_activations=True)),
    ]
    for tag, cfg in configs:
        manager = TRELLISQuantizationManager(
            model_name=model_name,
            output_dir=f"{base_output}/exp12_{tag}",
            modules_to_quantize=None,
            **cfg,
        )
        if manager.run_experiment():
            results.append(Path(manager.output_dir))
    return results


# 실험 3: G_L (slat_flow_model) 모듈만 INT8 양자화

def run_experiment_3(model_name: str = "text-base", base_output: str = "quantization_results") -> Path:
    manager = TRELLISQuantizationManager(
        model_name=model_name,
        output_dir=f"{base_output}/exp3_gl",
        modules_to_quantize=["slat_flow_model"],
        quantize_weights=True,
        quantize_activations=True,
    )
    manager.run_experiment()
    return Path(manager.output_dir)


# 실험 2: G_S (sparse_structure_flow_model) 모듈만 INT8 양자화

def run_experiment_2(model_name: str = "text-base", base_output: str = "quantization_results") -> Path:
    manager = TRELLISQuantizationManager(
        model_name=model_name,
        output_dir=f"{base_output}/exp2_gs",
        modules_to_quantize=["sparse_structure_flow_model"],
        quantize_weights=True,
        quantize_activations=True,
    )
    manager.run_experiment()
    return Path(manager.output_dir)


# 실험 10: 각 디코더별 INT8 양자화 후 품질 비교

def run_experiment_10(model_name: str = "text-base", base_output: str = "quantization_results") -> List[Path]:
    decoder_names = ["slat_decoder_gs", "slat_decoder_rf", "slat_decoder_mesh"]
    outputs = []
    for name in decoder_names:
        manager = TRELLISQuantizationManager(
            model_name=model_name,
            output_dir=f"{base_output}/exp10_{name}",
            modules_to_quantize=[name],
            quantize_weights=True,
            quantize_activations=True,
        )
        manager.run_experiment()
        outputs.append(Path(manager.output_dir))
    return outputs


__all__ = [
    "run_experiment_12",
    "run_experiment_3",
    "run_experiment_2",
    "run_experiment_10",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TRELLIS 양자화 실험 실행")
    parser.add_argument("experiment", choices=["12", "3", "2", "10"], help="실험 번호")
    parser.add_argument("--model", default="text-base", help="모델 이름")
    parser.add_argument("--output_dir", default="quantization_results", help="출력 디렉토리")
    args = parser.parse_args()

    if args.experiment == "12":
        run_experiment_12(args.model, args.output_dir)
    elif args.experiment == "3":
        run_experiment_3(args.model, args.output_dir)
    elif args.experiment == "2":
        run_experiment_2(args.model, args.output_dir)
    elif args.experiment == "10":
        run_experiment_10(args.model, args.output_dir)