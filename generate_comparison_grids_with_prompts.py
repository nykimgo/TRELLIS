#!/usr/bin/env python3
"""
Script to generate comparison grids for LLM-augmented prompt results in 3D asset generation.
Creates side-by-side comparisons for different model sizes and objects with prompts.
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import textwrap

# Base paths
BASE_PATH = "/mnt/nas/tmp/nayeon/output/CLIP_evaluation/TRELLIS-text-large"
GT_BASE_PATH = "/mnt/nas/Benchmark_Datatset/Toys4k/render_multiviews_for_CLIPeval"
OUTPUT_PATH = "/mnt/nas/tmp/nayeon/evaluation/fd_dinov2/TRELLIS-text-large"

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

# Target objects will be loaded dynamically from Excel file

def load_excel_data(excel_path):
    """Load and process Excel data for prompt information"""
    try:
        if not os.path.exists(excel_path):
            print(f"Excel file not found: {excel_path}")
            return None
        
        df = pd.read_excel(excel_path)
        print(f"Loaded Excel data from {excel_path} with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def get_target_objects_from_excel(df):
    """Extract target objects from Excel data"""
    if df is None:
        return []
    
    target_objects = set()
    
    for _, row in df.iterrows():
        # Extract middle part from file_identifier (e.g., "giraffe/giraffe_006/giraffe_006.blend" -> "giraffe_006")
        file_id = str(row['file_identifier'])
        if '/' in file_id:
            parts = file_id.split('/')
            if len(parts) >= 2:
                target_obj = parts[1]  # Get the middle part
                target_objects.add(target_obj)
    
    # Convert to sorted list
    target_list = sorted(list(target_objects))
    print(f"Found {len(target_list)} unique target objects from Excel")
    print("Sample objects:", target_list[:10])
    
    # Filter objects that actually exist in the file system
    existing_objects = []
    for obj in target_list:
        # Check if this object exists in any model directory
        exists = False
        for size_config in MODEL_CONFIGS.values():
            for model_name in size_config["models"]:
                model_path = os.path.join(BASE_PATH, model_name, obj)
                if os.path.exists(model_path):
                    exists = True
                    break
            if exists:
                break
        
        if exists:
            existing_objects.append(obj)
    
    print(f"Found {len(existing_objects)} objects that exist in the file system (out of {len(target_list)})")
    
    # For testing, limit to first 20 objects to avoid timeout
    limited_list = existing_objects[:20]
    print(f"Processing first {len(limited_list)} existing objects for testing")
    
    return limited_list

def get_prompt_info(df, llm_model, object_name):
    """Get prompt information for a specific model and object"""
    if df is None:
        return None, None
    
    # Extract object name and identifier from file_identifier (e.g., "giraffe_006")
    obj_name = object_name.split('_')[0]  # "giraffe"
    file_identifier = f'{obj_name}/{object_name}/{object_name}.blend'
    
    # Convert model name from directory format to Excel format
    excel_model_name = llm_model.replace('_', ':')
    
    # Find matching row by file_identifier middle part
    matches = df[
        (df['llm_model'] == excel_model_name) & 
        (df['file_identifier'] == file_identifier)
    ]
    
    if len(matches) > 0:
        row = matches.iloc[0]
        return row['user_prompt'], row['text_prompt']
    else:
        print(f"No prompt found for {excel_model_name} - {object_name} (converted from {llm_model})")
        return None, None

def wrap_text(text, width=80):
    """Wrap text for display"""
    if text is None:
        return "No prompt found"
    return '\n'.join(textwrap.wrap(str(text), width=width))

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

def create_comparison_grid_with_prompts(model_size, object_name, models, df):
    """Create comparison grid with prompts for a specific model size and object"""
    
    # Setup figure with more space for prompts
    fig = plt.figure(figsize=(24, 6 * (len(models) + 1)))
    
    # Main title - larger and closer
    main_title = f"LLM-Augmented Prompt 3D Asset Generation Results"
    fig.suptitle(main_title, fontsize=32, fontweight='bold', y=0.96)
    
    # Subtitle
    subtitle1 = MODEL_CONFIGS[model_size]["title"]
    subtitle2 = f"Object: {object_name}"
    
    # Create grid layout: (models + GT) x 5 (label + 4 views + prompt)
    rows = (len(models) + 1) * 2  # *2 for image and prompt rows
    cols = 5  # label + 4 views
    
    # Add GT row first
    gt_path = os.path.join(GT_BASE_PATH, object_name)
    
    # Get user prompt for GT
    user_prompt, _ = get_prompt_info(df, models[0], object_name)
    
    # GT images row
    row_idx = 0
    
    # GT label - much larger font
    ax = plt.subplot2grid((rows, cols), (row_idx, 0), fig=fig)
    ax.text(0.5, 0.5, 'Toys4k GT', ha='center', va='center', fontsize=25, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # GT images
    for col_idx, view_idx in enumerate([270, 315, 90, 135]):
        gt_img_path = os.path.join(gt_path, f"gt_view_{view_idx}.png")
        ax = plt.subplot2grid((rows, cols), (row_idx, col_idx + 1), fig=fig)
        
        img = load_image_safe(gt_img_path)
        if img is not None:
            ax.imshow(img)
        else:
            placeholder = create_placeholder_image()
            ax.imshow(placeholder)
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
        
        ax.axis('off')
    
    # GT prompt row
    ax = plt.subplot2grid((rows, cols), (row_idx + 1, 0), colspan=5, fig=fig)
    wrapped_prompt = wrap_text(user_prompt, width=100)
    ax.text(0.5, 0.5, f"User Prompt: {wrapped_prompt}", ha='center', va='center', 
            fontsize=24, fontweight='bold', wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Model rows
    for model_idx, model_name in enumerate(models):
        img_row_idx = (model_idx + 1) * 2
        prompt_row_idx = img_row_idx + 1
        
        # Check if this model has this object
        model_path = os.path.join(BASE_PATH, model_name, object_name)
        
        # Get prompts for this model
        user_prompt, text_prompt = get_prompt_info(df, model_name, object_name)
        
        # Model label - three lines with FD score
        ax = plt.subplot2grid((rows, cols), (img_row_idx, 0), fig=fig)
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
            ax.text(0.5, 0.75, line1, ha='center', va='center', fontsize=25, fontweight='bold')
            # Line 2: Size info (not bold) - closer spacing
            ax.text(0.5, 0.55, line2, ha='center', va='center', fontsize=25)
            # Line 3: FD score (bold) - larger gap
            ax.text(0.5, 0.25, fd_text, ha='center', va='center', fontsize=25, fontweight='bold')
        else:
            # Two lines only if no colon split
            ax.text(0.5, 0.65, line1, ha='center', va='center', fontsize=25, fontweight='bold')
            ax.text(0.5, 0.35, fd_text, ha='center', va='center', fontsize=25, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Model images
        for col_idx, view_idx in enumerate([270, 315, 90, 135]):
            gen_img_path = os.path.join(model_path, f"gen_view_{view_idx}.png")
            ax = plt.subplot2grid((rows, cols), (img_row_idx, col_idx + 1), fig=fig)
            
            img = load_image_safe(gen_img_path)
            if img is not None:
                ax.imshow(img)
            else:
                placeholder = create_placeholder_image()
                ax.imshow(placeholder)
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
            
            ax.axis('off')
        
        # Model prompt row
        ax = plt.subplot2grid((rows, cols), (prompt_row_idx, 0), colspan=5, fig=fig)
        wrapped_prompt = wrap_text(text_prompt, width=100)
        ax.text(0.5, 0.5, f"Enhanced Prompt: {wrapped_prompt}", ha='center', va='center', 
                fontsize=20, wrap=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Add subtitles - larger and closer together
    fig.text(0.5, 0.91, subtitle1, ha='center', va='center', fontsize=24, fontweight='bold')
    fig.text(0.5, 0.88, subtitle2, ha='center', va='center', fontsize=20)
    
    # Adjust layout - much tighter spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.005, wspace=0.005)
    
    # Save figure
    output_filename = f"{model_size}_{object_name}_with_prompts.png"
    output_path = os.path.join(OUTPUT_PATH, output_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path

def create_cross_size_comparison_by_family(object_name, df, model_family):
    """Create comparison across different model sizes for the same model family"""
    
    # Find models from the same family across different sizes
    family_models = {}
    for size, config in MODEL_CONFIGS.items():
        for model_name in config["models"]:
            # Extract model family name (e.g., "gemma3" from "gemma3_1b")
            model_base = model_name.split('_')[0].split('-')[0]
            if model_base == model_family:
                model_path = os.path.join(BASE_PATH, model_name, object_name)
                if os.path.exists(model_path):
                    # Check if this model has prompts available
                    excel_model_name = model_name.replace('_', ':')
                    obj_name = object_name.split('_')[0]  # "giraffe" from "giraffe_006"
                    obj_id = object_name.split('_')[1]    # "006" from "giraffe_006"
                    
                    matches = df[
                        (df['llm_model'] == excel_model_name) & 
                        (df['object_name_clean'] == obj_name) &
                        (df['sha256'].str.startswith(obj_id))
                    ]
                    
                    if len(matches) > 0:
                        family_models[size] = model_name
                        break
    
    if len(family_models) < 2:
        return None
    
    # Setup figure
    fig = plt.figure(figsize=(24, 6 * (len(family_models) + 1)))
    
    # Main title
    main_title = f"LLM-Augmented Prompt 3D Asset Generation Results"
    fig.suptitle(main_title, fontsize=32, fontweight='bold', y=0.96)
    
    # Subtitle
    subtitle1 = f"{model_family.title()} Model Size Comparison"
    subtitle2 = f"Object: {object_name.split('_')[0]}"
    
    # Create grid layout
    rows = (len(family_models) + 1) * 2  # *2 for image and prompt rows
    cols = 5
    
    # GT row
    gt_path = os.path.join(GT_BASE_PATH, object_name)
    user_prompt, _ = get_prompt_info(df, list(family_models.values())[0], object_name)
    
    # GT images
    row_idx = 0
    ax = plt.subplot2grid((rows, cols), (row_idx, 0), fig=fig)
    ax.text(0.5, 0.5, 'Toys4k GT', ha='center', va='center', fontsize=25, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    for col_idx, view_idx in enumerate([270, 315, 90, 135]):
        gt_img_path = os.path.join(gt_path, f"gt_view_{view_idx}.png")
        ax = plt.subplot2grid((rows, cols), (row_idx, col_idx + 1), fig=fig)
        
        img = load_image_safe(gt_img_path)
        if img is not None:
            ax.imshow(img)
        else:
            placeholder = create_placeholder_image()
            ax.imshow(placeholder)
            ax.text(0.5, 0.5, 'Missing', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12, color='red')
        ax.axis('off')
    
    # GT prompt
    ax = plt.subplot2grid((rows, cols), (row_idx + 1, 0), colspan=5, fig=fig)
    wrapped_prompt = wrap_text(user_prompt, width=100)
    ax.text(0.5, 0.5, f"User Prompt: {wrapped_prompt}", ha='center', va='center', 
            fontsize=24, wrap=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Model rows by size
    for size_idx, (size, model_name) in enumerate(family_models.items()):
        img_row_idx = (size_idx + 1) * 2
        prompt_row_idx = img_row_idx + 1
        
        model_path = os.path.join(BASE_PATH, model_name, object_name)
        user_prompt, text_prompt = get_prompt_info(df, model_name, object_name)
        
        # Model label with FD score
        ax = plt.subplot2grid((rows, cols), (img_row_idx, 0), fig=fig)
        display_name = model_name.replace('_', ':').replace('-', '-')
        size_label = f"{size.title()} Model"
        
        # Read FD score
        fd_score = read_fd_score(model_path)
        fd_text = f"FD: {fd_score}" if fd_score is not None else "FD: N/A"
        
        # Display three lines for cross-size comparison
        ax.text(0.5, 0.75, size_label, ha='center', va='center', fontsize=32, fontweight='bold')
        ax.text(0.5, 0.55, display_name, ha='center', va='center', fontsize=28)
        ax.text(0.5, 0.25, fd_text, ha='center', va='center', fontsize=28, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Model images
        for col_idx, view_idx in enumerate([270, 315, 90, 135]):
            gen_img_path = os.path.join(model_path, f"gen_view_{view_idx}.png")
            ax = plt.subplot2grid((rows, cols), (img_row_idx, col_idx + 1), fig=fig)
            
            img = load_image_safe(gen_img_path)
            if img is not None:
                ax.imshow(img)
            else:
                placeholder = create_placeholder_image()
                ax.imshow(placeholder)
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, color='red')
            ax.axis('off')
        
        # Model prompt
        ax = plt.subplot2grid((rows, cols), (prompt_row_idx, 0), colspan=5, fig=fig)
        wrapped_prompt = wrap_text(text_prompt, width=100)
        ax.text(0.5, 0.5, f"Enhanced Prompt: {wrapped_prompt}", ha='center', va='center', 
                fontsize=20, wrap=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # Add subtitles
    fig.text(0.5, 0.91, subtitle1, ha='center', va='center', fontsize=24, fontweight='bold')
    fig.text(0.5, 0.88, subtitle2, ha='center', va='center', fontsize=20)
    
    # Adjust layout - much tighter spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, hspace=0.005, wspace=0.005)
    
    # Save figure
    output_filename = f"cross_size_{object_name}_comparison_{model_family}.png"
    output_path = os.path.join(OUTPUT_PATH, output_filename)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path

def main():
    """Main function to generate all comparison grids"""
    
    print("Starting comparison grid generation with prompts...")
    
    # Get Excel file path from user
    if len(sys.argv) > 1:
        excel_path = sys.argv[1]
    else:
        excel_path = input("Please enter the path to the Excel file: ").strip()
        if not excel_path:
            print("No Excel file path provided. Exiting.")
            return
    
    print(f"Using Excel file: {excel_path}")
    
    # Load Excel data
    df = load_excel_data(excel_path)
    if df is None:
        print("Failed to load Excel data. Exiting.")
        return
    
    # Get target objects from Excel
    target_objects = get_target_objects_from_excel(df)
    if not target_objects:
        print("No target objects found in Excel data. Exiting.")
        return
    
    generated_files = []
    
    # Generate grids for each model size and object combination (with prompts)
    print("\n=== Generating size-based comparisons with prompts ===")
    for model_size, config in MODEL_CONFIGS.items():
        print(f"\nProcessing {model_size} models...")
        
        for object_name in target_objects:
            print(f"  Processing object: {object_name}")
            
            # Check which models have this object
            available_models = []
            for model_name in config["models"]:
                model_path = os.path.join(BASE_PATH, model_name, object_name)
                if os.path.exists(model_path):
                    available_models.append(model_name)
            
            if available_models:
                # Check if all models in this size category are available
                expected_models = config["models"]
                if len(available_models) < len(expected_models):
                    missing_models = set(expected_models) - set(available_models)
                    print(f"    Available models: {available_models}")
                    print(f"    Missing models: {list(missing_models)} - skipping {object_name}")
                    continue
                
                # Check if all models have prompts available
                all_prompts_found = True
                for model_name in available_models:
                    excel_model_name = model_name.replace('_', ':')
                    obj_name = object_name.split('_')[0]  # "giraffe" from "giraffe_006"
                    file_identifier = f'{obj_name}/{object_name}/{object_name}.blend'
                    matches = df[
                        (df['llm_model'] == excel_model_name) & 
                        (df['file_identifier'] == file_identifier) 
                    ]
                    
                    if len(matches) == 0:
                        print(f"    No prompt found for {excel_model_name} - skipping {object_name}")
                        all_prompts_found = False
                        break
                
                if all_prompts_found:
                    print(f"    Available models: {available_models}")
                    output_file = create_comparison_grid_with_prompts(model_size, object_name, available_models, df)
                    if output_file:
                        generated_files.append(output_file)
                else:
                    print(f"    Skipping {object_name} due to missing prompts")
            else:
                print(f"    No models found for {object_name} in {model_size} category - skipping")
    
    # Generate cross-size comparisons by model family
    print("\n=== Generating cross-size comparisons by model family ===")
    
    # Extract unique model families from all available models
    model_families = set()
    for config in MODEL_CONFIGS.values():
        for model_name in config["models"]:
            family = model_name.split('_')[0].split('-')[0]
            model_families.add(family)
    
    for object_name in target_objects:
        print(f"Processing cross-size comparison for: {object_name}")
        for family in sorted(model_families):
            print(f"  Checking {family} family...")
            output_file = create_cross_size_comparison_by_family(object_name, df, family)
            if output_file:
                generated_files.append(output_file)
                print(f"    Created {family} family comparison")
            else:
                print(f"    Not enough {family} models across sizes - skipping")
    
    print(f"\n✅ Generation complete! Created {len(generated_files)} comparison grids.")
    print("\nGenerated files:")
    for file_path in generated_files:
        print(f"  - {file_path}")

if __name__ == "__main__":
    print("LLM-Augmented Prompt 3D Asset Generation Comparison Tool")
    print("=" * 60)
    print("Usage:")
    print("  python generate_comparison_grids_with_prompts.py [excel_file_path]")
    print("  or run without arguments to enter path interactively")
    print("=" * 60)
    main()