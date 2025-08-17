#!/usr/bin/env python3
"""
Test script for FD_dinov2 Evaluator

This script performs basic functionality tests to ensure the evaluator
can properly load data, render assets, extract features, and calculate FD.
"""

import os
import pandas as pd
import torch
from fd_dinov2_evaluator import FDDinoV2Evaluator

def test_dinov2_loading():
    """Test DINOv2 model loading"""
    print("Testing DINOv2 model loading...")
    
    try:
        evaluator = FDDinoV2Evaluator()
        print(f"✓ DINOv2 model loaded successfully on {evaluator.device}")
        print(f"✓ Model type: {type(evaluator.dinov2_model)}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading DINOv2 model: {e}")
        return False

def test_data_loading():
    """Test loading sampled data and LLM results"""
    print("\nTesting data loading...")
    
    sampled_csv = "/mnt/nas/tmp/nayeon/sampled_data_100_random.csv"
    results_excel = "/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx"
    
    if not os.path.exists(sampled_csv):
        print(f"❌ Sampled CSV not found: {sampled_csv}")
        return False
    
    if not os.path.exists(results_excel):
        print(f"❌ Results Excel not found: {results_excel}")
        return False
    
    try:
        evaluator = FDDinoV2Evaluator()
        
        sampled_df = evaluator.load_sampled_data(sampled_csv)
        llm_df = evaluator.load_llm_results(results_excel)
        
        print(f"✓ Loaded {len(sampled_df)} sampled assets")
        print(f"✓ Loaded {len(llm_df)} LLM results")
        
        # Check for matching sha256s
        common_sha256s = set(sampled_df['sha256']).intersection(set(llm_df['sha256']))
        print(f"✓ Found {len(common_sha256s)} common assets between datasets")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_asset_finding():
    """Test finding assets (both ground truth and generated)"""
    print("\nTesting asset file discovery...")
    
    try:
        evaluator = FDDinoV2Evaluator()
        dataset_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
        output_base_path = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
        
        # Test ground truth asset
        test_file_identifier = "giraffe/giraffe_006/giraffe_006.blend"
        gt_asset = evaluator.find_asset_file(dataset_path, test_file_identifier)
        
        if gt_asset:
            print(f"✓ Found ground truth asset: {os.path.basename(gt_asset)}")
        else:
            print(f"⚠ Ground truth asset not found for: {test_file_identifier}")
        
        # Test generated asset
        gen_assets = evaluator.find_generated_assets(output_base_path, 'gemma3', 'Giraffe')
        
        if gen_assets:
            print(f"✓ Found {len(gen_assets)} generated assets for gemma3/Giraffe")
            print(f"   Example: {os.path.basename(gen_assets[0])}")
        else:
            print(f"⚠ No generated assets found for gemma3/Giraffe")
        
        return gt_asset is not None or len(gen_assets) > 0
        
    except Exception as e:
        print(f"❌ Error finding assets: {e}")
        return False

def test_mesh_loading_and_rendering():
    """Test mesh loading and 4-view rendering"""
    print("\nTesting mesh loading and rendering...")
    
    try:
        evaluator = FDDinoV2Evaluator()
        output_base_path = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
        
        # Find a test asset
        gen_assets = evaluator.find_generated_assets(output_base_path, 'gemma3', 'Giraffe')
        if not gen_assets:
            print("⚠ No test assets available for mesh loading test")
            return False
        
        # Load mesh
        mesh = evaluator.load_3d_asset(gen_assets[0])
        if mesh is None:
            print(f"❌ Failed to load mesh from {gen_assets[0]}")
            return False
        
        print(f"✓ Loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Test 4-view rendering
        rendered_images = evaluator.render_asset_4views(mesh)
        
        if len(rendered_images) == 4:
            print(f"✓ Successfully rendered {len(rendered_images)} views")
            print(f"✓ Image resolution: {rendered_images[0].shape}")
            return True
        else:
            print(f"❌ Expected 4 views, got {len(rendered_images)}")
            return False
            
    except Exception as e:
        print(f"❌ Error in mesh loading/rendering: {e}")
        return False

def test_feature_extraction():
    """Test DINOv2 feature extraction"""
    print("\nTesting DINOv2 feature extraction...")
    
    try:
        evaluator = FDDinoV2Evaluator()
        output_base_path = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
        
        # Get test asset and render
        gen_assets = evaluator.find_generated_assets(output_base_path, 'gemma3', 'Giraffe')
        if not gen_assets:
            print("⚠ No test assets available for feature extraction test")
            return False
        
        mesh = evaluator.load_3d_asset(gen_assets[0])
        if mesh is None:
            print("❌ Failed to load test mesh")
            return False
        
        rendered_images = evaluator.render_asset_4views(mesh)
        if len(rendered_images) != 4:
            print("❌ Failed to render test images")
            return False
        
        # Extract features
        features = evaluator.extract_dinov2_features(rendered_images)
        
        if features.shape[0] == 4 and features.shape[1] == 1024:
            print(f"✓ Extracted features with shape: {features.shape}")
            print(f"✓ Feature range: [{features.min().item():.3f}, {features.max().item():.3f}]")
            return True
        else:
            print(f"❌ Unexpected feature shape: {features.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        return False

def test_frechet_distance():
    """Test Fréchet Distance calculation"""
    print("\nTesting Fréchet Distance calculation...")
    
    try:
        evaluator = FDDinoV2Evaluator()
        
        # Create dummy features for testing
        torch.manual_seed(42)
        features_real = torch.randn(4, 1024)
        features_fake = torch.randn(4, 1024) + 0.1  # Slightly different distribution
        
        fd_value = evaluator.calculate_frechet_distance(features_real, features_fake)
        
        print(f"✓ Calculated FD value: {fd_value:.4f}")
        
        # Test with identical distributions (should be close to 0)
        fd_identical = evaluator.calculate_frechet_distance(features_real, features_real)
        print(f"✓ FD for identical distributions: {fd_identical:.6f}")
        
        if 0 <= fd_value < 10000 and fd_identical < 1e-2:
            return True
        else:
            print(f"❌ Unexpected FD values")
            return False
            
    except Exception as e:
        print(f"❌ Error in FD calculation: {e}")
        return False

def test_minimal_evaluation():
    """Test minimal end-to-end evaluation"""
    print("\nTesting minimal evaluation pipeline...")
    
    try:
        evaluator = FDDinoV2Evaluator()
        
        sampled_csv = "/mnt/nas/tmp/nayeon/sampled_data_100_random.csv"
        results_excel = "/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx"
        dataset_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
        output_base_path = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
        
        # Run evaluation with just 1 asset pair
        results = evaluator.evaluate_fd_dinov2(
            sampled_csv_path=sampled_csv,
            results_excel_path=results_excel,
            dataset_path=dataset_path,
            output_base_path=output_base_path,
            save_path="test_fd_minimal.csv",
            max_assets=1
        )
        
        print(f"✓ Minimal evaluation completed")
        print(f"✓ Success rate: {results['success_rate']:.1%}")
        print(f"✓ FD_dinov2: {results['mean_fd_dinov2']:.4f}")
        
        # Clean up test file
        if os.path.exists("test_fd_minimal.csv"):
            os.remove("test_fd_minimal.csv")
        if os.path.exists("test_fd_minimal_summary.json"):
            os.remove("test_fd_minimal_summary.json")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in minimal evaluation: {e}")
        return False

def main():
    """Run all tests"""
    print("=== FD_dinov2 Evaluator Test Suite ===\n")
    
    tests = [
        ("DINOv2 Loading", test_dinov2_loading),
        ("Data Loading", test_data_loading),
        ("Asset Finding", test_asset_finding),
        ("Mesh Loading & Rendering", test_mesh_loading_and_rendering),
        ("Feature Extraction", test_feature_extraction),
        ("Fréchet Distance", test_frechet_distance),
        ("Minimal Evaluation", test_minimal_evaluation)
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
        print("🎉 All tests passed! The FD_dinov2 evaluator is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()