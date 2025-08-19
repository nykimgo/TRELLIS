import pandas as pd

df = pd.read_excel('/mnt/nas/tmp/nayeon/sampled_data_100_random_results.xlsx')

print('Looking for target objects:')
targets = ['Apple', 'Cake', 'Hammer', 'Keyboard', 'Reindeer', 'Robot', 'Truck']

for target in targets:
    matches = df[df['object_name_clean'] == target]
    if len(matches) > 0:
        print(f'{target}: {len(matches)} matches, sha256: {matches.iloc[0]["sha256"][:6]}')
    else:
        print(f'{target}: No matches found')

print('\nAvailable objects:')
print(sorted(df['object_name_clean'].unique()))