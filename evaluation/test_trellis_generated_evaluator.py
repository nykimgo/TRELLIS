#!/usr/bin/env python3
"""
Test script for TRELLIS Generated CLIP Score Evaluator

This script performs basic functionality tests to ensure the evaluator
can properly load data, find assets, and run evaluations.
"""

import os
import pandas as pd
from trellis_generated_clip_evaluator import TrellisGeneratedCLIPEvaluator

def test_data_loading():
    """Test loading LLM results Excel file"""
    print("Testing data loading...")
    
    results_path = "/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx"
    
    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return False
    
    try:
        evaluator = TrellisGeneratedCLIPEvaluator()
        df = evaluator.load_llm_results(results_path)
        
        print(f"✓ Loaded {len(df)} entries from Excel file")
        print(f"✓ Columns: {list(df.columns)}")
        print(f"✓ LLM models found: {df['category'].unique().tolist()}")
        print(f"✓ Sample object names: {df['object_name_clean'].head(3).tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_asset_finding():
    """Test finding generated 3D assets"""
    print("\nTesting asset file discovery...")
    
    try:
        evaluator = TrellisGeneratedCLIPEvaluator()
        output_base_path = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
        
        # Test a few known combinations
        test_cases = [
            ('gemma3', 'Giraffe'),
            ('gemma3', 'Pig'),
            ('gemma3', 'Dinosaur')
        ]
        
        found_assets = 0
        for llm_model, object_name in test_cases:
            assets = evaluator.find_generated_assets(output_base_path, llm_model, object_name)
            if assets:
                print(f"✓ Found {len(assets)} assets for {llm_model}/{object_name}")
                print(f"   Example: {os.path.basename(assets[0])}")
                found_assets += 1
            else:
                print(f"⚠ No assets found for {llm_model}/{object_name}")
        
        if found_assets > 0:
            print(f"✓ Asset discovery working ({found_assets}/{len(test_cases)} test cases)")
            return True
        else:
            print("❌ No assets found in any test cases")
            return False
            
    except Exception as e:
        print(f"❌ Error finding assets: {e}")
        return False

def test_mesh_loading():
    """Test loading 3D mesh files"""
    print("\nTesting 3D mesh loading...")
    
    try:
        evaluator = TrellisGeneratedCLIPEvaluator()
        output_base_path = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
        
        # Find first available asset
        assets = evaluator.find_generated_assets(output_base_path, 'gemma3', 'Giraffe')
        if not assets:
            print("⚠ No test assets available for mesh loading test")
            return False
        
        asset_path = assets[0]
        mesh = evaluator.load_3d_asset(asset_path)
        
        if mesh is not None:
            print(f"✓ Successfully loaded mesh from {os.path.basename(asset_path)}")
            print(f"✓ Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
            return True
        else:
            print(f"❌ Failed to load mesh from {asset_path}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading mesh: {e}")
        return False

def test_clip_model():
    """Test CLIP model initialization"""
    print("\nTesting CLIP model initialization...")
    
    try:
        evaluator = TrellisGeneratedCLIPEvaluator()
        
        # Test text feature extraction
        test_text = ["A 3D giraffe with a long neck"]
        text_features = evaluator.extract_text_features(test_text)
        
        print(f"✓ CLIP model loaded successfully")
        print(f"✓ Text features shape: {text_features.shape}")
        print(f"✓ Running on device: {evaluator.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with CLIP model: {e}")
        return False

def test_full_evaluation():
    """Test full evaluation with minimal data"""
    print("\nTesting full evaluation pipeline...")
    
    try:
        evaluator = TrellisGeneratedCLIPEvaluator()
        
        results_path = "/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx"
        output_base_path = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
        
        # Run evaluation with just 1 asset
        results = evaluator.evaluate_trellis_generated_dataset(
            results_excel_path=results_path,
            output_base_path=output_base_path,
            save_path="test_minimal_evaluation.csv",
            max_assets=1
        )
        
        print(f"✓ Full evaluation completed")
        print(f"✓ Success rate: {results['success_rate']:.1%}")
        print(f"✓ CLIP Score: {results['mean_clip_score_scaled']:.2f}")
        
        # Clean up test file
        if os.path.exists("test_minimal_evaluation.csv"):
            os.remove("test_minimal_evaluation.csv")
        if os.path.exists("test_minimal_evaluation_summary.json"):
            os.remove("test_minimal_evaluation_summary.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in full evaluation: {e}")
        return False

def main():
    """Run all tests"""
    print("=== TRELLIS Generated CLIP Evaluator Test Suite ===\n")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Asset Finding", test_asset_finding),
        ("Mesh Loading", test_mesh_loading),
        ("CLIP Model", test_clip_model),
        ("Full Evaluation", test_full_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        if test_func():
            passed += 1
            print(f"✓ {test_name} PASSED\n")
        else:
            print(f"❌ {test_name} FAILED\n")
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 All tests passed! The evaluator is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()