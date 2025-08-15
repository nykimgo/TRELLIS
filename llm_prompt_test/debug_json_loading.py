import json

# Load the JSON file and check structure
with open('prompt_generation_outputs/parsed_json_from_groq.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("JSON structure:")
print(f"Metadata: {data['metadata']}")
print(f"Number of entries: {len(data['normalized_data'])}")

print("\nFirst few entries:")
for i, entry in enumerate(data['normalized_data'][:3]):
    print(f"Entry {i+1}:")
    for key, value in entry.items():
        print(f"  {key}: {value}")
    print()

print("\nDeepSeek entries:")
deepseek_entries = [entry for entry in data['normalized_data'] if 'deepseek' in entry.get('llm_model', '').lower()]
for i, entry in enumerate(deepseek_entries):
    print(f"DeepSeek Entry {i+1}:")
    for key, value in entry.items():
        print(f"  {key}: {value}")
    print()