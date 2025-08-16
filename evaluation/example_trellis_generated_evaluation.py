#!/usr/bin/env python3
"""
Example usage of TRELLIS Generated CLIP Score Evaluator

This script demonstrates how to use the TrellisGeneratedCLIPEvaluator
for evaluating CLIP scores between TRELLIS-generated 3D assets and 
their LLM-augmented text prompts.
"""

import os
from trellis_generated_clip_evaluator import TrellisGeneratedCLIPEvaluator

def main():
    print("=== TRELLIS Generated CLIP Score Evaluation Example ===\n")
    
    # Paths to your data
    results_excel_path = "/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx"
    output_base_path = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
    
    # Initialize evaluator
    print("1. Initializing CLIP evaluator...")
    evaluator = TrellisGeneratedCLIPEvaluator(clip_model_name="openai/clip-vit-base-patch32")
    print("✓ Evaluator initialized\n")
    
    # Example 1: Quick test with limited assets
    print("2. Running quick test (max 5 assets)...")
    test_results = evaluator.evaluate_trellis_generated_dataset(
        results_excel_path=results_excel_path,
        output_base_path=output_base_path,
        save_path="test_trellis_clip_scores.csv",
        max_assets=5
    )
    print(f"✓ Test completed - CLIP Score: {test_results['mean_clip_score_scaled']:.2f}\n")
    
    # Example 2: Evaluate specific LLM models only
    print("3. Evaluating specific LLM models (gemma3, qwen3)...")
    model_results = evaluator.evaluate_trellis_generated_dataset(
        results_excel_path=results_excel_path,
        output_base_path=output_base_path,
        save_path="specific_models_clip_scores.csv",
        llm_models_filter=['gemma3', 'qwen3'],
        max_assets=10
    )
    print(f"✓ Model evaluation completed - CLIP Score: {model_results['mean_clip_score_scaled']:.2f}\n")
    
    # Example 3: Compare results across models
    print("4. Model Performance Comparison:")
    for model, summary in model_results['model_summaries'].items():
        print(f"   {model:12s}: {summary['mean_clip_score_scaled']:6.2f} "
              f"({summary['successful_evaluations']:2d}/{summary['total_assets']:2d} assets)")
    
    print("\n=== Evaluation Complete ===")
    print(f"Results saved to: test_trellis_clip_scores.csv")
    print(f"Model results saved to: specific_models_clip_scores.csv")


if __name__ == "__main__":
    main()