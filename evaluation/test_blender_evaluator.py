#!/usr/bin/env python3
"""
Test script for the Blender-based FD_dinov2 evaluator
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

# Add the evaluation directory to Python path
sys.path.append('/home/sr/TRELLIS/evaluation')

from fd_dinov2_blender_evaluator import FD_dinov2_BlenderEvaluator

def create_test_csv():
    """Create a test CSV file with a few sample assets"""
    test_data = [
        {'uid': '000_elephant'},
        {'uid': '001_chair'}, 
        {'uid': '002_car'}
    ]
    
    test_df = pd.DataFrame(test_data)
    test_csv_path = '/tmp/test_sample.csv'
    test_df.to_csv(test_csv_path, index=False)
    return test_csv_path

def main():
    print("Testing Blender-based FD_dinov2 evaluator...")
    
    # Create test CSV
    test_csv = create_test_csv()
    print(f"Created test CSV: {test_csv}")
    
    # Test directories (using sample Toys4k data)
    generated_dir = "/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816"
    ground_truth_dir = "/mnt/nas/Benchmark_Datatset/Toys4k/meshes"
    
    # Initialize evaluator
    try:
        evaluator = FD_dinov2_BlenderEvaluator(device='cuda')
        print("✓ Evaluator initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize evaluator: {e}")
        return
    
    # Test Blender installation
    if not os.path.exists(evaluator.blender_path):
        print("Installing Blender for testing...")
        evaluator._install_blender()
    
    # Test single asset evaluation (if assets exist)
    test_assets = [
        "000_elephant.glb",
        "001_chair.glb", 
        "002_car.glb"
    ]
    
    found_asset = None
    for asset in test_assets:
        gen_path = os.path.join(generated_dir, asset)
        gt_path = os.path.join(ground_truth_dir, asset.replace('.glb', '.ply'))
        
        if os.path.exists(gen_path) and os.path.exists(gt_path):
            found_asset = (gen_path, gt_path, asset)
            break
    
    if found_asset:
        gen_path, gt_path, asset_name = found_asset
        print(f"Testing single asset evaluation with: {asset_name}")
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                result = evaluator.evaluate_single_asset(
                    gen_path, gt_path, 
                    save_dir=os.path.join(temp_dir, 'test_renders')
                )
                print(f"✓ Single asset evaluation result: {result}")
        except Exception as e:
            print(f"✗ Single asset evaluation failed: {e}")
    else:
        print("No matching test assets found for single evaluation")
    
    # Test rendering function
    test_model = None
    for asset in test_assets:
        test_path = os.path.join(generated_dir, asset)
        if os.path.exists(test_path):
            test_model = test_path
            break
    
    if test_model:
        print(f"Testing Blender rendering with: {os.path.basename(test_model)}")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                rendered_images = evaluator.render_4_views_blender(test_model, temp_dir)
                print(f"✓ Rendered {len(rendered_images)} views")
                for i, img_path in enumerate(rendered_images):
                    if os.path.exists(img_path):
                        size = os.path.getsize(img_path)
                        print(f"  View {i}: {img_path} ({size} bytes)")
                    else:
                        print(f"  View {i}: Missing - {img_path}")
        except Exception as e:
            print(f"✗ Blender rendering test failed: {e}")
    else:
        print("No test model found for rendering test")
    
    # Test feature extraction
    try:
        # Create dummy images for feature extraction test
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp_dir:
            dummy_images = []
            for i in range(4):
                # Create a simple colored image
                img = Image.new('RGB', (512, 512), color=(50 + i*50, 100, 150))
                img_path = os.path.join(temp_dir, f'test_{i}.png')
                img.save(img_path)
                dummy_images.append(img_path)
            
            features = evaluator.extract_dinov2_features(dummy_images)
            print(f"✓ Feature extraction test: {features.shape}")
            
            # Test Fréchet Distance calculation
            features2 = features + 0.1 * torch.randn_like(features)
            fd_score = evaluator.calculate_frechet_distance(features, features2)
            print(f"✓ Fréchet Distance test: {fd_score:.4f}")
            
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
    
    print("Testing completed!")

if __name__ == "__main__":
    main()