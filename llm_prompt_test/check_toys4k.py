#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Load Toys4k dataset
df = pd.read_csv('/mnt/nas/Benchmark_Datatset/Toys4k/metadata.csv')

print('Total rows:', len(df))
print('Captions column info:')
print('Type:', df['captions'].dtype)
print('Null count:', df['captions'].isnull().sum())
print('Float count:', df['captions'].apply(lambda x: isinstance(x, float)).sum())
print()

print('Sample captions types:')
for i, caption in enumerate(df['captions'].head(10)):
    print(f'Row {i}: {type(caption).__name__} - {str(caption)[:100]}')

# Check for problematic rows
null_rows = df[df['captions'].isnull()]
if len(null_rows) > 0:
    print(f'\nFound {len(null_rows)} null captions rows:')
    print(null_rows[['sha256', 'captions']].head())

float_rows = df[df['captions'].apply(lambda x: isinstance(x, float))]
if len(float_rows) > 0:
    print(f'\nFound {len(float_rows)} float captions rows:')
    print(float_rows[['sha256', 'captions']].head())