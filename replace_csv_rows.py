#!/usr/bin/env python3

import pandas as pd
import sys

def replace_false_with_true_rows():
    # File paths
    original_file = "/mnt/nas/tmp/nayeon/evaluation/CLIP_evaluation_sampled_data_100_random_results.csv"
    replacement_file = "/mnt/nas/tmp/nayeon/evaluation/CLIP_evaluation_CLIP_evaluation_sampled_data_100_random_results-FALSE.csv"
    
    print("Reading original file...")
    df_original = pd.read_csv(original_file)
    
    print("Reading replacement file...")
    df_replacement = pd.read_csv(replacement_file)
    
    # Find FALSE rows in original file
    false_rows = df_original[df_original['success'] == False]
    print(f"Found {len(false_rows)} FALSE rows in original file")
    
    # Find TRUE rows in replacement file
    true_rows = df_replacement[df_replacement['success'] == True]
    print(f"Found {len(true_rows)} TRUE rows in replacement file")
    
    if len(true_rows) == 0:
        print("No TRUE rows found in replacement file!")
        return
    
    # Create a copy of the original dataframe for modification
    df_modified = df_original.copy()
    
    replacements_made = 0
    
    # For each FALSE row in original, try to find a matching TRUE row in replacement by sha256
    for idx, false_row in false_rows.iterrows():
        sha256_value = false_row['sha256']
        
        # Skip if sha256 is empty or NaN
        if pd.isna(sha256_value) or sha256_value == '':
            print(f"Row {idx}: No sha256 value, skipping")
            continue
            
        # Find matching row in replacement file by sha256
        matches = true_rows[true_rows['sha256'] == sha256_value]
        
        if len(matches) > 0:
            # Replace only specific fields, keeping original structure
            replacement_row = matches.iloc[0]
            print(f"Replacing row {idx} with sha256='{sha256_value[:16]}...'")
            
            # Update only the fields that should change (success-related fields)
            df_modified.loc[idx, 'clip_score'] = replacement_row['clip_score']
            df_modified.loc[idx, 'num_views_rendered'] = replacement_row['num_views_rendered']
            df_modified.loc[idx, 'success'] = replacement_row['success']
            df_modified.loc[idx, 'asset_path'] = replacement_row['asset_path']
            # Clear error field if success is True
            if replacement_row['success']:
                df_modified.loc[idx, 'error'] = ''
            
            replacements_made += 1
        else:
            print(f"No match found for row {idx} with sha256='{sha256_value[:16]}...'")
    
    print(f"\nTotal replacements made: {replacements_made}")
    
    # Save the modified file
    output_file = "/mnt/nas/tmp/nayeon/evaluation/CLIP_evaluation_sampled_data_100_random_results_updated.csv"
    df_modified.to_csv(output_file, index=False)
    print(f"Updated file saved to: {output_file}")
    
    # Show summary
    original_success_count = len(df_original[df_original['success'] == True])
    modified_success_count = len(df_modified[df_modified['success'] == True])
    
    print(f"\nSummary:")
    print(f"Original file - TRUE success: {original_success_count}, FALSE success: {len(df_original) - original_success_count}")
    print(f"Modified file - TRUE success: {modified_success_count}, FALSE success: {len(df_modified) - modified_success_count}")
    print(f"Success rate improved from {original_success_count/len(df_original)*100:.1f}% to {modified_success_count/len(df_modified)*100:.1f}%")

if __name__ == "__main__":
    replace_false_with_true_rows()