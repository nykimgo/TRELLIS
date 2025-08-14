#!/usr/bin/env python3
"""
Simple test for CLIP Score Evaluator components
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import trimesh
from pathlib import Path

def test_mesh_loading():
    """Test mesh loading functionality."""
    print("Testing mesh loading...")
    
    # Test dataset path
    toys4k_path = "/mnt/nas/Benchmark_Datatset/Toys4k"
    
    if not os.path.exists(toys4k_path):
        print(f"✗ Toys4k dataset not found at: {toys4k_path}")
        return False
    
    print(f"✓ Toys4k dataset found at: {toys4k_path}")
    
    # Find sample 3D assets
    asset_extensions = ['.glb', '.ply', '.obj', '.gltf']
    asset_files = []
    
    for ext in asset_extensions:
        asset_files.extend(Path(toys4k_path).glob(f'**/*{ext}'))
    
    print(f"✓ Found {len(asset_files)} 3D assets in dataset")
    
    if len(asset_files) == 0:
        print("✗ No 3D assets found!")
        return False
    
    # Test loading a few assets
    successful_loads = 0
    for i, asset_file in enumerate(asset_files[:5]):  # Test first 5 assets
        try:
            mesh = trimesh.load(str(asset_file))
            
            # Handle scene objects
            if isinstance(mesh, trimesh.Scene):
                meshes = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
                if meshes:
                    mesh = meshes[0]
                else:
                    continue
            
            if isinstance(mesh, trimesh.Trimesh):
                print(f"✓ Loaded {asset_file.name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                successful_loads += 1
            else:
                print(f"✗ Invalid mesh type for {asset_file.name}")
                
        except Exception as e:
            print(f"✗ Failed to load {asset_file.name}: {e}")
    
    print(f"✓ Successfully loaded {successful_loads}/5 test assets")
    return successful_loads > 0

def test_rendering():
    """Test basic rendering functionality."""
    print("\nTesting rendering functionality...")
    
    try:
        # Create a simple test mesh (cube)
        mesh = trimesh.creation.box(extents=(1, 1, 1))
        print(f"✓ Created test cube: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Test matplotlib rendering
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Simple mesh visualization
        vertices = mesh.vertices
        faces = mesh.faces
        
        ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       triangles=faces, alpha=0.8, color='lightblue')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Test Mesh Rendering')
        
        # Save test image
        plt.savefig('test_render.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        if os.path.exists('test_render.png'):
            print("✓ Rendering test successful - saved test_render.png")
            os.remove('test_render.png')  # Clean up
            return True
        else:
            print("✗ Rendering test failed - no output file")
            return False
            
    except Exception as e:
        print(f"✗ Rendering test failed: {e}")
        return False

def test_viewpoint_calculation():
    """Test viewpoint calculation."""
    print("\nTesting viewpoint calculation...")
    
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
        
        print(f"  View {i+1}: yaw={yaw}°, position=({x:.2f}, {y:.2f}, {z:.2f})")
    
    return True

def main():
    """Main test function."""
    print("CLIP Score Evaluator Component Tests")
    print("=" * 50)
    
    tests = [
        ("Mesh Loading", test_mesh_loading),
        ("Rendering", test_rendering),
        ("Viewpoint Calculation", test_viewpoint_calculation),
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
    
    if passed == total:
        print("\n✓ All component tests passed!")
        print("\nTo run full CLIP evaluation (requires CLIP model):")
        print("python clip_score_evaluator.py --dataset_path /mnt/nas/Benchmark_Datatset/Toys4k --output_path results.csv")
    else:
        print(f"\n✗ {total - passed} tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()