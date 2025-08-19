#!/usr/bin/env python3

import pandas as pd

def check_model_distribution():
    print("Reading files...")
    df_original = pd.read_csv('/mnt/nas/tmp/nayeon/evaluation/CLIP_evaluation_sampled_data_100_random_results.csv')
    df_updated = pd.read_csv('/mnt/nas/tmp/nayeon/evaluation/CLIP_evaluation_sampled_data_100_random_results_updated.csv')
    
    print('Original file model distribution:')
    print(df_original['llm_model'].value_counts().sort_index())
    print(f'Total original: {len(df_original)}')
    
    print('\nUpdated file model distribution:')
    print(df_updated['llm_model'].value_counts().sort_index())
    print(f'Total updated: {len(df_updated)}')
    
    print('\nOriginal success distribution:')
    print(df_original['success'].value_counts())
    
    print('\nUpdated success distribution:')
    print(df_updated['success'].value_counts())

if __name__ == "__main__":
    check_model_distribution()