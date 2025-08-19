#!/usr/bin/env python3
"""
Script to generate comparison grids for LLM-augmented prompt results in 3D asset generation.
Creates side-by-side comparisons for different model sizes and objects.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np

# Base paths
BASE_PATH = "/mnt/nas/tmp/nayeon/evaluation/fd_dinov2_blender_with_save_renders/TRELLIS-text-large"
OUTPUT_PATH = "/mnt/nas/tmp/nayeon/evaluation/fd_dinov2_blender_with_save_renders/TRELLIS-text-large"

# Model configurations
MODEL_CONFIGS = {
    "small": {
        "models": ["gemma3_1b", "qwen3_0.6b", "deepseek-r1_1.5b", "llama3.1_8b"],
        "title": "Small (GPU Usage 1GB ~ 6GB) LLM models"
    },
    "medium": {
        "models": ["qwen3_14b", "gemma3_12b", "gpt-oss_20b", "deepseek-r1_14b-qwen-distill-q8_0"],
        "title": "Medium (GPU Usage 10GB ~ 16GB) LLM models"
    },
    "large": {
        "models": ["gemma3_27b-it-q8_0", "deepseek-r1_32b-qwen-distill-q8_0", "qwen3_32b-q8_0", "llama3.1_70b-instruct-q4_0"],
        "title": "Large (GPU Usage 30GB ~ 40GB) LLM models"
    }
}

# Target objects
TARGET_OBJECTS = [
    "Apple_316018",
    "Cake_48e95a", 
    "Hammer_373df1",
    "Keyboard_7e3e99",
    "Reindeer_f2d2ae",
    "Robot_9ea7ed",
    "Truck_f025a8"
]

def load_image_safe(image_path):
    """Load image if exists, otherwise return None"""
    if os.path.exists(image_path):
        try:
            return mpimg.imread(image_path)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    return None

def read_fd_score(model_path):
    """Read FD_dinov2 score from feature_stats.txt if available"""
    stats_file = os.path.join(model_path, "feature_stats.txt")
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if 'FD_dinov2 score:' in line:
                        score_str = line.split('FD_dinov2 score:')[1].strip()
                        try:
                            score = float(score_str)
                            return round(score, 2)
                        except ValueError:
                            return None
        except Exception as e:
            print(f"Error reading FD score from {stats_file}: {e}")
    return None

def create_placeholder_image(width=512, height=512):
    """Create a placeholder image for missing files"""
    placeholder = np.ones((height, width, 3)) * 0.9  # Light gray
    return placeholder

def create_comparison_grid(model_size, object_name, models):
    """Create comparison grid for a specific model size and object"""
    
    # Setup figure with tighter layout
    fig = plt.figure(figsize=(20, 4 * (len(models) + 1)))
    
    # Main title - larger and closer
    main_title = f"LLM-Augmented Prompt 3D Asset Generation Results"
    fig.suptitle(main_title, fontsize=32, fontweight='bold', y=0.96)
    
    # Subtitle
    subtitle1 = MODEL_CONFIGS[model_size]["title"]
    subtitle2 = f"Object: {object_name.split('_')[0]}"
    
    # Create grid layout: (models + GT) x 5 (label + 4 views)
    rows = len(models) + 1  # +1 for GT row
    cols = 5  # label + 4 views
    
    # Add GT row first
    gt_path = os.path.join(BASE_PATH, models[0], object_name)  # Use first available model's GT
    
    # GT row
    row_idx = 0
    
    # GT label - larger font
    ax = plt.subplot2grid((rows, cols), (row_idx, 0), fig=fig)
    ax.text(0.5, 0.5, 'Toys4k GT', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # GT images
    for view_idx in range(4):
        gt_img_path = os.path.join(gt_path, f"gt_view_{view_idx}.png")
        ax = plt.subplot2grid((rows, cols), (row_idx, view_idx + 1), fig=fig)
        
        img = load_image_safe(gt_img_path)
        if img is not None:
            ax.imshow(img)
        else:
            placeholder = create_placeholder_image()
            ax.imshow(placeholder)
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
        
        ax.axis('off')
    
    # Model rows
    for model_idx, model_name in enumerate(models):
        row_idx = model_idx + 1
        
        # Check if this model has this object
        model_path = os.path.join(BASE_PATH, model_name, object_name)
        
        # Model label - three lines with FD score
        ax = plt.subplot2grid((rows, cols), (row_idx, 0), fig=fig)
        display_name = model_name.replace('_', ':').replace('-', '-')
        
        # Split model name at colon for better display
        if ':' in display_name:
            parts = display_name.split(':', 1)
            line1 = parts[0]  # Model name (bold)
            line2 = ':' + parts[1]  # Size info (not bold)
        else:
            line1 = display_name
            line2 = ''
        
        # Read FD score
        fd_score = read_fd_score(model_path)
        fd_text = f"FD: {fd_score}" if fd_score is not None else "FD: N/A"
        
        # Display three lines with appropriate spacing and styling
        if line2:
            # Line 1: Model name (bold)
            ax.text(0.5, 0.75, line1, ha='center', va='center', fontsize=18, fontweight='bold')
            # Line 2: Size info (not bold) - closer spacing
            ax.text(0.5, 0.55, line2, ha='center', va='center', fontsize=18)
            # Line 3: FD score (bold) - larger gap
            ax.text(0.5, 0.25, fd_text, ha='center', va='center', fontsize=18, fontweight='bold')
        else:
            # Two lines only if no colon split
            ax.text(0.5, 0.65, line1, ha='center', va='center', fontsize=18, fontweight='bold')
            ax.text(0.5, 0.35, fd_text, ha='center', va='center', fontsize=18, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Model images
        for view_idx in range(4):
            gen_img_path = os.path.join(model_path, f"gen_view_{view_idx}.png")
            ax = plt.subplot2grid((rows, cols), (row_idx, view_idx + 1), fig=fig)
            
            img = load_image_safe(gen_img_path)
            if img is not None:
                ax.imshow(img)
            else:
                placeholder = create_placeholder_image()
                ax.imshow(placeholder)
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
            
            ax.axis('off')
    
    # Add subtitles - larger and closer together
    fig.text(0.5, 0.91, subtitle1, ha='center', va='center', fontsize=24, fontweight='bold')
    fig.text(0.5, 0.88, subtitle2, ha='center', va='center', fontsize=20)
    
    # Adjust layout - tighter spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.05, wspace=0.05)
    
    # Save figure
    output_filename = f"{model_size}_{object_name}.png"
    output_path = os.path.join(OUTPUT_PATH, output_filename)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path

def main():
    """Main function to generate all comparison grids"""
    
    print("Starting comparison grid generation...")
    
    generated_files = []
    
    # Generate grids for each model size and object combination
    for model_size, config in MODEL_CONFIGS.items():
        print(f"\nProcessing {model_size} models...")
        
        for object_name in TARGET_OBJECTS:
            print(f"  Processing object: {object_name}")
            
            # Check which models have this object
            available_models = []
            for model_name in config["models"]:
                model_path = os.path.join(BASE_PATH, model_name, object_name)
                if os.path.exists(model_path):
                    available_models.append(model_name)
            
            if available_models:
                print(f"    Available models: {available_models}")
                output_file = create_comparison_grid(model_size, object_name, available_models)
                generated_files.append(output_file)
            else:
                print(f"    No models found for {object_name} in {model_size} category")
    
    print(f"\n✅ Generation complete! Created {len(generated_files)} comparison grids.")
    print("\nGenerated files:")
    for file_path in generated_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    main()