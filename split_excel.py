import pandas as pd

# Read the deduplicated Excel file
df = pd.read_excel('/mnt/nas/tmp/nayeon/sampled_data_100_random_results_rm_du.xlsx')

print(f'Total rows to split: {len(df)}')

# Split the data
df_01 = df.iloc[:400]  # First 400 rows
df_02 = df.iloc[400:800]  # Next 400 rows (rows 400-799)
df_03 = df.iloc[800:]  # Remaining rows (from row 800 onwards)

print(f'File 01: {len(df_01)} rows')
print(f'File 02: {len(df_02)} rows') 
print(f'File 03: {len(df_03)} rows')
print(f'Total check: {len(df_01) + len(df_02) + len(df_03)} (should equal {len(df)})')

# Save each split to separate Excel files
files_info = [
    (df_01, '/mnt/nas/tmp/nayeon/sampled_data_100_random_results_rm_du_01.xlsx'),
    (df_02, '/mnt/nas/tmp/nayeon/sampled_data_100_random_results_rm_du_02.xlsx'),
    (df_03, '/mnt/nas/tmp/nayeon/sampled_data_100_random_results_rm_du_03.xlsx')
]

for i, (data, filename) in enumerate(files_info, 1):
    data.to_excel(filename, index=False)
    print(f'✓ File {i:02d} saved: {filename} ({len(data)} rows)')

print('\n✓ All files split and saved successfully!')