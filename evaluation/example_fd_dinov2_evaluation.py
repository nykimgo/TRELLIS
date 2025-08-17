#!/usr/bin/env python3
"""
Example usage of FD_dinov2 Evaluator

This script demonstrates how to use the FDDinoV2Evaluator
for evaluating visual quality of TRELLIS-generated 3D assets.
"""

import os
from fd_dinov2_evaluator import FDDinoV2Evaluator

def main():
    print("=== FD_dinov2 Evaluation Examples ===\n")
    
    # Default paths - you can change these to your data
    sampled_csv = "/mnt/nas/tmp/nayeon/sampled_data_100_random.csv"
    results_excel = "/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx"
    dataset_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    output_base_path = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
    
    print("Example 1: Quick Test with 3 Asset Pairs")
    print("-" * 50)
    
    # Initialize evaluator
    print("1. Initializing FD_dinov2 evaluator...")
    evaluator = FDDinoV2Evaluator()
    print("✓ Evaluator initialized\n")
    
    # Quick test with limited assets
    print("2. Running quick test (max 3 asset pairs)...")
    test_results = evaluator.evaluate_fd_dinov2(
        sampled_csv_path=sampled_csv,
        results_excel_path=results_excel,
        dataset_path=dataset_path,
        output_base_path=output_base_path,
        save_path="test_fd_dinov2_results.csv",
        max_assets=3
    )
    print(f"✓ Test completed - FD_dinov2: {test_results['mean_fd_dinov2']:.4f}\n")
    
    print("Example 2: Evaluate Specific LLM Models")
    print("-" * 50)
    
    # Evaluate specific models only
    print("3. Evaluating specific LLM models (gemma3, qwen3)...")
    model_results = evaluator.evaluate_fd_dinov2(
        sampled_csv_path=sampled_csv,
        results_excel_path=results_excel,
        dataset_path=dataset_path,
        output_base_path=output_base_path,
        save_path="model_fd_dinov2_results.csv",
        llm_models_filter=['gemma3', 'qwen3'],
        max_assets=5
    )
    print(f"✓ Model evaluation completed - FD_dinov2: {model_results['mean_fd_dinov2']:.4f}\n")
    
    print("Example 3: Model Performance Comparison")
    print("-" * 50)
    
    print("4. Model Performance Comparison:")
    for model, summary in model_results['model_summaries'].items():
        print(f"   {model:12s}: FD_dinov2 = {summary['mean_fd_dinov2']:8.4f} "
              f"({summary['successful_evaluations']:2d}/{summary['total_assets']:2d} pairs)")
    
    print("\n=== Key Insights ===")
    print("• Lower FD_dinov2 values indicate better visual quality")
    print("• FD_dinov2 measures how similar the generated asset distribution is to real assets")
    print("• Values typically range from 0 (identical) to several hundred (very different)")
    
    print("\n=== Files Created ===")
    print("  - test_fd_dinov2_results.csv (3 asset pairs, all models)")
    print("  - model_fd_dinov2_results.csv (5 asset pairs, gemma3 + qwen3 only)")
    
    print("\n=== Command Line Usage Examples ===")
    print("You can also run these evaluations directly from command line:")
    print()
    print("# Quick test:")
    print("python fd_dinov2_evaluator.py --max_assets 5 --save_path test_fd.csv")
    print()
    print("# Specific models:")
    print("python fd_dinov2_evaluator.py \\")
    print("  --llm_models gemma3 qwen3 \\")
    print("  --save_path model_comparison_fd.csv")
    print()
    print("# Full evaluation:")
    print("python fd_dinov2_evaluator.py --save_path full_fd_dinov2_evaluation.csv")
    print()
    print("# Use different Excel file:")
    print("python fd_dinov2_evaluator.py \\")
    print("  --results_excel /path/to/your/results_part02.xlsx \\")
    print("  --save_path part02_fd_results.csv")


if __name__ == "__main__":
    main()