import pandas as pd

df = pd.read_excel('/mnt/nas/tmp/nayeon/sampled_data_100_random_results.xlsx')

# Check specific examples
target_objects = [
    "Apple_316018",
    "Cake_48e95a",
    "Hammer_373df1"
]

for target in target_objects:
    obj_name = target.split('_')[0]
    obj_id = target.split('_')[1]
    
    print(f"\nLooking for {target}:")
    print(f"  Object name: {obj_name}")
    print(f"  Identifier: {obj_id}")
    
    # Check if object name exists
    name_matches = df[df['object_name_clean'] == obj_name]
    print(f"  Objects with name '{obj_name}': {len(name_matches)}")
    
    if len(name_matches) > 0:
        print("  Available SHA256 prefixes for this object:")
        for _, row in name_matches.head(5).iterrows():
            sha_prefix = row['sha256'][:6]
            print(f"    {sha_prefix} ({row['llm_model']})")
        
        # Check for exact match
        exact_matches = name_matches[name_matches['sha256'].str.startswith(obj_id)]
        print(f"  Exact matches with prefix '{obj_id}': {len(exact_matches)}")
        
        if len(exact_matches) > 0:
            print("  Found exact matches:")
            for _, row in exact_matches.iterrows():
                print(f"    {row['llm_model']}: {row['sha256'][:6]}")