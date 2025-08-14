#!/usr/bin/env python3
"""
Test script for CLIP Score Evaluator
"""

import os
import sys
from clip_score_evaluator import CLIPScoreEvaluator

def test_clip_evaluator():
    """Test the CLIP Score evaluator with a sample setup."""
    
    print("Testing CLIP Score Evaluator...")
    
    # Initialize evaluator
    try:
        evaluator = CLIPScoreEvaluator()
        print("✓ CLIP Score Evaluator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize evaluator: {e}")
        return False
    
    # Test text feature extraction
    try:
        test_prompts = ["a red car", "a blue house", "a green tree"]
        text_features = evaluator.extract_text_features(test_prompts)
        print(f"✓ Text feature extraction works. Shape: {text_features.shape}")
    except Exception as e:
        print(f"✗ Text feature extraction failed: {e}")
        return False
    
    # Test dataset path
    toys4k_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    if os.path.exists(toys4k_path):
        print(f"✓ Toys4k dataset found at: {toys4k_path}")
        
        # List some files to verify structure
        try:
            import glob
            asset_files = []
            for ext in ['*.glb', '*.ply', '*.obj', '*.gltf']:
                asset_files.extend(glob.glob(os.path.join(toys4k_path, '**', ext), recursive=True))
            
            print(f"✓ Found {len(asset_files)} 3D assets in dataset")
            if asset_files:
                print(f"  Sample assets: {asset_files[:3]}")
        except Exception as e:
            print(f"✗ Error scanning dataset: {e}")
    else:
        print(f"✗ Toys4k dataset not found at: {toys4k_path}")
        print("  Please verify the dataset path is correct")
    
    print("\nCLIP Score Evaluator test completed!")
    return True

def main():
    """Main test function."""
    success = test_clip_evaluator()
    
    if success:
        print("\n" + "="*50)
        print("To run CLIP Score evaluation on Toys4k dataset:")
        print("python clip_score_evaluator.py --dataset_path /mnt/nas/Benchmark_Datatset/Toys4k --output_path toys4k_clip_scores.csv")
        print("\nOr with a prompts file:")
        print("python clip_score_evaluator.py --dataset_path /mnt/nas/Benchmark_Datatset/Toys4k --prompts_file prompts.json --output_path toys4k_clip_scores.csv")
    else:
        print("\n" + "="*50)
        print("Test failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()