#!/usr/bin/env python3

import pandas as pd
import json

def update_summary_json():
    # File paths
    csv_file = "/mnt/nas/tmp/nayeon/evaluation/CLIP_evaluation_sampled_data_100_random_results_updated.csv"
    json_file = "/mnt/nas/tmp/nayeon/evaluation/CLIP_evaluation_sampled_data_100_random_results_summary.json"
    
    print("Reading updated CSV file...")
    df = pd.read_csv(csv_file)
    
    print("Reading existing JSON file...")
    with open(json_file, 'r') as f:
        summary = json.load(f)
    
    # Calculate overall statistics
    total_assets = len(df)
    successful_evaluations = len(df[df['success'] == True])
    success_rate = successful_evaluations / total_assets if total_assets > 0 else 0.0
    
    # Calculate CLIP scores for successful evaluations only
    successful_df = df[df['success'] == True]
    if len(successful_df) > 0:
        mean_clip_score = successful_df['clip_score'].mean()
        mean_clip_score_scaled = mean_clip_score * 100  # Scale to 0-100
    else:
        mean_clip_score = 0.0
        mean_clip_score_scaled = 0.0
    
    print(f"Overall statistics:")
    print(f"  Total assets: {total_assets}")
    print(f"  Successful evaluations: {successful_evaluations}")
    print(f"  Success rate: {success_rate:.3f}")
    print(f"  Mean CLIP score: {mean_clip_score:.6f}")
    print(f"  Mean CLIP score (scaled): {mean_clip_score_scaled:.3f}")
    
    # Update overall statistics
    summary["total_assets"] = total_assets
    summary["successful_evaluations"] = successful_evaluations
    summary["success_rate"] = success_rate
    summary["mean_clip_score"] = mean_clip_score
    summary["mean_clip_score_scaled"] = mean_clip_score_scaled
    
    # Calculate per-model statistics
    print("\nModel-specific statistics:")
    for model in summary["model_summaries"].keys():
        model_df = df[df['llm_model'] == model]
        
        if len(model_df) > 0:
            model_total = len(model_df)
            model_successful = len(model_df[model_df['success'] == True])
            model_success_rate = model_successful / model_total if model_total > 0 else 0.0
            
            # Calculate CLIP scores for successful evaluations only
            model_successful_df = model_df[model_df['success'] == True]
            if len(model_successful_df) > 0:
                model_mean_clip = model_successful_df['clip_score'].mean()
                model_mean_clip_scaled = model_mean_clip * 100
            else:
                model_mean_clip = 0.0
                model_mean_clip_scaled = 0.0
            
            print(f"  {model}:")
            print(f"    Total: {model_total}, Successful: {model_successful}, Success rate: {model_success_rate:.3f}")
            print(f"    Mean CLIP score: {model_mean_clip:.6f}, Scaled: {model_mean_clip_scaled:.3f}")
            
            # Update model summary
            summary["model_summaries"][model]["total_assets"] = model_total
            summary["model_summaries"][model]["successful_evaluations"] = model_successful
            summary["model_summaries"][model]["success_rate"] = model_success_rate
            summary["model_summaries"][model]["mean_clip_score"] = model_mean_clip
            summary["model_summaries"][model]["mean_clip_score_scaled"] = model_mean_clip_scaled
        else:
            print(f"  {model}: No data found")
    
    # Save updated JSON
    print(f"\nSaving updated summary to {json_file}")
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Summary JSON file updated successfully!")

if __name__ == "__main__":
    update_summary_json()