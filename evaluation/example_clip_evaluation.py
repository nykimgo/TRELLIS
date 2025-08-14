#!/usr/bin/env python3
"""
Example usage of CLIP Score Evaluator for TRELLIS results
"""

import os
import json
from clip_score_evaluator import CLIPScoreEvaluator

def create_sample_prompts_file():
    """Create a sample prompts file for demonstration."""
    sample_prompts = {
        "toy_car_01": "a red toy car with four wheels",
        "toy_house_02": "a small wooden house with windows",
        "toy_animal_03": "a cute stuffed bear toy",
        "toy_plane_04": "a blue airplane toy with wings",
        "toy_train_05": "a colorful toy train with carriages"
    }
    
    # Save as JSON
    with open('sample_prompts.json', 'w') as f:
        json.dump(sample_prompts, f, indent=2)
    
    # Also save as CSV format
    import pandas as pd
    df = pd.DataFrame([
        {'asset_name': k, 'prompt': v} for k, v in sample_prompts.items()
    ])
    df.to_csv('sample_prompts.csv', index=False)
    
    print("Created sample prompts files: sample_prompts.json and sample_prompts.csv")

def run_clip_evaluation_example():
    """Run CLIP Score evaluation example."""
    
    print("CLIP Score Evaluation Example for TRELLIS")
    print("=" * 50)
    
    # Dataset path
    toys4k_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    
    if not os.path.exists(toys4k_path):
        print(f"Toys4k dataset not found at: {toys4k_path}")
        print("Please verify the dataset path and try again.")
        return
    
    # Initialize evaluator
    print("Initializing CLIP Score Evaluator...")
    evaluator = CLIPScoreEvaluator(clip_model_name="openai/clip-vit-base-patch32")
    
    # Example 1: Evaluate single asset
    print("\n1. Single Asset Evaluation Example")
    print("-" * 30)
    
    # Find a sample asset file
    import glob
    sample_assets = []
    for ext in ['*.glb', '*.ply', '*.obj']:
        sample_assets.extend(glob.glob(os.path.join(toys4k_path, '**', ext), recursive=True))
    
    if sample_assets:
        sample_asset = sample_assets[0]
        sample_prompt = "a colorful toy object"
        
        print(f"Evaluating: {os.path.basename(sample_asset)}")
        print(f"Prompt: {sample_prompt}")
        
        result = evaluator.evaluate_single_asset(sample_asset, sample_prompt)
        
        if result['success']:
            print(f"✓ CLIP Score: {result['clip_score']:.4f}")
            print(f"✓ Views rendered: {result['num_views_rendered']}")
        else:
            print("✗ Evaluation failed")
    else:
        print("No sample assets found for single evaluation")
    
    # Example 2: Evaluate dataset (limited to first 5 assets for demo)
    print("\n2. Dataset Evaluation Example (first 5 assets)")
    print("-" * 30)
    
    # Limit to first 5 assets for demo
    limited_assets = sample_assets[:5] if len(sample_assets) >= 5 else sample_assets
    
    if limited_assets:
        results = []
        total_score = 0
        successful = 0
        
        for asset_path in limited_assets:
            # Use filename as prompt (in practice, load from metadata)
            filename = os.path.basename(asset_path)
            prompt = filename.replace('_', ' ').replace('-', ' ').split('.')[0]
            
            result = evaluator.evaluate_single_asset(asset_path, prompt)
            results.append(result)
            
            print(f"Asset: {filename[:30]:<30} | Prompt: {prompt[:20]:<20} | Score: {result['clip_score']:.4f if result['success'] else 'FAILED'}")
            
            if result['success']:
                total_score += result['clip_score']
                successful += 1
        
        if successful > 0:
            mean_score = total_score / successful
            print(f"\nSummary:")
            print(f"  Successful evaluations: {successful}/{len(limited_assets)}")
            print(f"  Mean CLIP Score: {mean_score:.4f}")
            print(f"  Mean CLIP Score (×100): {mean_score * 100:.2f}")
        
        # Save results
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv('example_clip_results.csv', index=False)
        print(f"  Results saved to: example_clip_results.csv")
    
    # Example 3: Using prompts file
    print("\n3. Evaluation with Prompts File Example")
    print("-" * 30)
    
    create_sample_prompts_file()
    
    print("To run full evaluation with prompts file:")
    print("python clip_score_evaluator.py --dataset_path /mnt/nas/Benchmark_Datatset/Toys4k --prompts_file sample_prompts.json --output_path full_results.csv")
    
    print("\nTo run full dataset evaluation:")
    print("python clip_score_evaluator.py --dataset_path /mnt/nas/Benchmark_Datatset/Toys4k --output_path full_dataset_results.csv")

def main():
    """Main function."""
    try:
        run_clip_evaluation_example()
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()