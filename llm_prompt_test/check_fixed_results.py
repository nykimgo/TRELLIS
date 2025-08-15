#!/usr/bin/env python3
import pandas as pd

df = pd.read_excel("prompt_generation_outputs/automated_prompt_results.xlsx")

print("=== FIXES VERIFICATION ===")
print(f"Total rows: {len(df)}")
print()

print("=== 1. Object Names by Model (Fixed qwen3 'Unknown' issue) ===")
for model in df["llm_model"].unique():
    objects = list(df[df["llm_model"]==model]["object_name"].unique())
    print(f"{model}: {objects}")
print()

print("=== 2. SHA256 Status (Fixed qwen3 empty fields) ===")
for model in df["llm_model"].unique():
    model_data = df[df["llm_model"]==model]
    has_sha256 = len(model_data[model_data["sha256"] != ""])
    total = len(model_data)
    print(f"{model}: {has_sha256}/{total} have sha256")
print()

print("=== 3. Text Prompt Formatting ===")
for model in df["llm_model"].unique():
    sample = df[df["llm_model"]==model].iloc[0]
    text = sample["text_prompt"]
    print(f"{model}:")
    print(f"  Sample: {text[:100]}...")
    starts_with_quote = text.startswith('"')
    has_number_prefix = any(text.startswith(f"{i}.") for i in range(1, 10))
    print(f"  Starts with quote: {starts_with_quote}")
    print(f"  Has number prefix: {has_number_prefix}")
    print()

print("=== Summary of Fixes ===")
qwen3_data = df[df["llm_model"]=="qwen3:0.6b"]
print(f"1. qwen3 object_name 'Unknown' fixed: {list(qwen3_data['object_name'].unique())}")
print(f"2. qwen3 sha256 filled: {len(qwen3_data[qwen3_data['sha256'] != ''])}/{len(qwen3_data)} entries")
qwen3_sample = qwen3_data.iloc[0]["text_prompt"]
quote_char = '"'
print(f"3. qwen3 quotes added: starts with quote = {qwen3_sample.startswith(quote_char)}")

gemma3_data = df[df["llm_model"]=="gemma3:1b"]
gemma3_sample = gemma3_data.iloc[0]["text_prompt"]
has_number = any(gemma3_sample.startswith(f"{i}.") for i in range(1, 10))
print(f"4. gemma3 number prefix removed: has number prefix = {has_number}")

deepseek_data = df[df["llm_model"]=="deepseek-r1:1.5b"]
deepseek_sample = deepseek_data.iloc[0]["text_prompt"]
print(f"5. deepseek quotes added: starts with quote = {deepseek_sample.startswith(quote_char)}")