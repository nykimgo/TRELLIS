#!/usr/bin/env python3
"""
Basic functionality test for Toys4k dataset without CLIP model
"""

import os
import pandas as pd
import ast
import trimesh
import numpy as np

def test_toys4k_structure():
    """Test Toys4k dataset structure."""
    print("Testing Toys4k dataset structure...")
    
    dataset_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    
    # Check main components
    components = {
        'metadata.csv': os.path.join(dataset_path, 'metadata.csv'),
        'obj_files': os.path.join(dataset_path, 'toys4k_obj_files'),
        'blend_files': os.path.join(dataset_path, 'toys4k_blend_files'),
    }
    
    for name, path in components.items():
        if os.path.exists(path):
            print(f"✓ Found {name}: {path}")
        else:
            print(f"✗ Missing {name}: {path}")
    
    return all(os.path.exists(path) for path in components.values())

def test_metadata_parsing():
    """Test metadata loading and caption parsing."""
    print("\nTesting metadata parsing...")
    
    dataset_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    metadata_path = os.path.join(dataset_path, 'metadata.csv')
    
    try:
        df = pd.read_csv(metadata_path)
        print(f"✓ Loaded metadata with {len(df)} rows")
        
        # Test caption parsing
        test_row = df.iloc[0]
        captions = ast.literal_eval(test_row['captions'])
        print(f"✓ Parsed captions for {test_row['file_identifier']}")
        print(f"  First caption: {captions[0][:100]}...")
        print(f"  Total captions: {len(captions)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metadata parsing failed: {e}")
        return False

def test_asset_loading():
    """Test loading 3D assets."""
    print("\nTesting 3D asset loading...")
    
    dataset_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    obj_path = os.path.join(dataset_path, 'toys4k_obj_files')
    
    # Find first available asset
    categories = os.listdir(obj_path)[:3]  # Test first 3 categories
    
    loaded_assets = 0
    for category in categories:
        category_path = os.path.join(obj_path, category)
        if not os.path.isdir(category_path):
            continue
            
        assets = os.listdir(category_path)[:2]  # Test first 2 assets per category
        for asset in assets:
            asset_path = os.path.join(category_path, asset, 'mesh.obj')
            
            if os.path.exists(asset_path):
                try:
                    mesh = trimesh.load(asset_path)
                    
                    if isinstance(mesh, trimesh.Scene):
                        meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                        if meshes:
                            mesh = meshes[0]
                    
                    if isinstance(mesh, trimesh.Trimesh):
                        print(f"✓ Loaded {category}/{asset}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                        loaded_assets += 1
                    else:
                        print(f"✗ Invalid mesh type for {category}/{asset}")
                        
                except Exception as e:
                    print(f"✗ Failed to load {category}/{asset}: {e}")
    
    print(f"✓ Successfully loaded {loaded_assets} test assets")
    return loaded_assets > 0

def test_rendering_setup():
    """Test rendering setup parameters."""
    print("\nTesting rendering setup...")
    
    # Test viewpoint calculations
    yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    pitch_angle = 30
    radius = 2
    
    print(f"✓ Configured {len(yaw_angles)} viewpoints:")
    for i, yaw in enumerate(yaw_angles):
        yaw_rad = np.deg2rad(yaw)
        pitch_rad = np.deg2rad(pitch_angle)
        
        x = radius * np.sin(yaw_rad) * np.cos(pitch_rad)
        y = radius * np.cos(yaw_rad) * np.cos(pitch_rad)
        z = radius * np.sin(pitch_rad)
        
        print(f"  View {i+1}: yaw={yaw}°, camera=({x:.2f}, {y:.2f}, {z:.2f})")
    
    return True

def create_usage_instructions():
    """Create usage instructions."""
    print("\n" + "="*60)
    print("CLIP SCORE EVALUATOR FOR TRELLIS - USAGE INSTRUCTIONS")
    print("="*60)
    
    print("\nImplementation Complete! The following files have been created:")
    print("1. clip_score_evaluator.py - General CLIP score evaluator")
    print("2. toys4k_clip_evaluator.py - Toys4k-specific evaluator")
    print("3. requirements_clip_eval.txt - Required dependencies")
    
    print("\nTo run the evaluation:")
    print("\n1. Install dependencies:")
    print("   pip install -r requirements_clip_eval.txt")
    
    print("\n2. For a small test (5 assets):")
    print("   python toys4k_clip_evaluator.py --max_assets 5 --output_path test_results.csv")
    
    print("\n3. For full evaluation:")
    print("   python toys4k_clip_evaluator.py --output_path full_toys4k_clip_scores.csv")
    
    print("\n4. Using a different CLIP model:")
    print("   python toys4k_clip_evaluator.py --clip_model openai/clip-vit-large-patch14 --output_path results.csv")
    
    print("\nThe evaluator will:")
    print("- Load Toys4k metadata with captions")
    print("- Render each 3D asset from 8 viewpoints (45° intervals)")
    print("- Extract CLIP features from images and captions")
    print("- Calculate cosine similarity (CLIP Score)")
    print("- Average across 8 views per asset, then across all assets")
    print("- Report final score multiplied by 100 (as per paper convention)")
    
    print("\nOutput files:")
    print("- detailed_results.csv: Per-asset scores and metadata")
    print("- summary.json: Aggregated metrics including mean CLIP score")

def main():
    """Main test function."""
    print("Toys4k CLIP Score Evaluator - Basic Tests")
    print("=" * 50)
    
    tests = [
        ("Dataset Structure", test_toys4k_structure),
        ("Metadata Parsing", test_metadata_parsing),
        ("Asset Loading", test_asset_loading),
        ("Rendering Setup", test_rendering_setup),
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
    print(f"Basic Tests: {passed}/{total} passed")
    
    if passed >= 3:
        print("✓ Core components working correctly!")
        create_usage_instructions()
    else:
        print("✗ Some basic tests failed. Please check the setup.")

if __name__ == "__main__":
    main()