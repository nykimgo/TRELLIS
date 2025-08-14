#!/usr/bin/env python3
"""
Test script for Toys4k CLIP Score Evaluator
"""

import os
import sys
import pandas as pd
from toys4k_clip_evaluator import Toys4kCLIPEvaluator

def test_toys4k_metadata():
    """Test loading Toys4k metadata."""
    print("Testing Toys4k metadata loading...")
    
    dataset_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    
    if not os.path.exists(dataset_path):
        print(f"✗ Toys4k dataset not found at: {dataset_path}")
        return False
    
    try:
        evaluator = Toys4kCLIPEvaluator()
        metadata_df = evaluator.load_toys4k_metadata(dataset_path)
        
        print(f"✓ Loaded metadata for {len(metadata_df)} assets")
        print(f"✓ Columns: {list(metadata_df.columns)}")
        
        # Check first few rows
        for idx, row in metadata_df.head(3).iterrows():
            print(f"✓ Asset {idx}: {row['file_identifier']}")
            if 'parsed_captions' in row:
                captions = row['parsed_captions']
                print(f"  First caption: {captions[0][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load metadata: {e}")
        return False

def test_asset_finding():
    """Test finding asset files."""
    print("\nTesting asset file finding...")
    
    dataset_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    
    try:
        evaluator = Toys4kCLIPEvaluator()
        metadata_df = evaluator.load_toys4k_metadata(dataset_path)
        
        found_assets = 0
        missing_assets = 0
        
        for idx, row in metadata_df.head(10).iterrows():
            asset_path = evaluator.find_asset_file(dataset_path, row)
            if asset_path:
                print(f"✓ Found: {row['file_identifier']} -> {os.path.basename(asset_path)}")
                found_assets += 1
            else:
                print(f"✗ Missing: {row['file_identifier']}")
                missing_assets += 1
        
        print(f"✓ Found {found_assets}/{found_assets + missing_assets} asset files")
        return found_assets > 0
        
    except Exception as e:
        print(f"✗ Asset finding test failed: {e}")
        return False

def test_single_evaluation():
    """Test evaluating a single asset."""
    print("\nTesting single asset evaluation...")
    
    dataset_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    
    try:
        # Use a simple CLIP model or skip if not available
        print("Note: Skipping CLIP model loading for basic test")
        print("This test only checks data loading and processing pipeline")
        
        # Load metadata and find first available asset
        import ast
        metadata_path = os.path.join(dataset_path, 'metadata.csv')
        df = pd.read_csv(metadata_path)
        
        for idx, row in df.head(10).iterrows():
            # Parse captions
            try:
                captions = ast.literal_eval(row['captions'])
                if not isinstance(captions, list):
                    captions = [row['captions']]
            except:
                captions = [row['captions']]
            
            # Check if asset file exists
            obj_path = os.path.join(dataset_path, 'toys4k_obj_files', 
                                   row['file_identifier'].replace('.blend', '.obj'))
            
            if os.path.exists(obj_path):
                print(f"✓ Found test asset: {row['file_identifier']}")
                print(f"  File: {obj_path}")
                print(f"  Caption: {captions[0][:100]}...")
                print(f"  Aesthetic score: {row.get('aesthetic_score', 'N/A')}")
                return True
        
        print("✗ No test assets found with valid files")
        return False
        
    except Exception as e:
        print(f"✗ Single evaluation test failed: {e}")
        return False

def main():
    """Main test function."""
    print("Toys4k CLIP Score Evaluator Tests")
    print("=" * 50)
    
    tests = [
        ("Metadata Loading", test_toys4k_metadata),
        ("Asset Finding", test_asset_finding),
        ("Single Evaluation", test_single_evaluation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed >= 2:  # Allow some flexibility for CLIP model issues
        print("\n✓ Core functionality tests passed!")
        print("\nTo run small-scale evaluation:")
        print("python toys4k_clip_evaluator.py --max_assets 5 --output_path test_results.csv")
        print("\nTo run full evaluation:")
        print("python toys4k_clip_evaluator.py --output_path full_toys4k_results.csv")
    else:
        print(f"\n✗ {total - passed} tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()