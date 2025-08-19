#!/usr/bin/env python3
"""
Debug script to understand the difference between use_fast=True and use_fast=False in CLIPProcessor
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def test_clip_processor(use_fast=True):
    """Test CLIP processor with use_fast parameter"""
    print(f"\n=== Testing with use_fast={use_fast} ===")
    
    # Load model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=use_fast)
    
    print(f"Processor class: {processor.__class__.__name__}")
    print(f"Tokenizer class: {processor.tokenizer.__class__.__name__}")
    print(f"Feature extractor class: {processor.feature_extractor.__class__.__name__}")
    
    # Test with simple text and image
    test_text = "a photo of a cat"
    # Create a simple test image (red square)
    test_image = Image.fromarray(np.full((224, 224, 3), [255, 0, 0], dtype=np.uint8))
    
    # Process text
    text_inputs = processor(text=[test_text], return_tensors="pt", padding=True, truncation=True)
    print(f"Text input keys: {list(text_inputs.keys())}")
    print(f"Text input_ids shape: {text_inputs['input_ids'].shape}")
    print(f"Text input_ids sample: {text_inputs['input_ids'][0][:10]}")
    
    # Process image
    image_inputs = processor(images=[test_image], return_tensors="pt", padding=True)
    print(f"Image input keys: {list(image_inputs.keys())}")
    print(f"Image pixel_values shape: {image_inputs['pixel_values'].shape}")
    print(f"Image pixel_values range: [{image_inputs['pixel_values'].min().item():.3f}, {image_inputs['pixel_values'].max().item():.3f}]")
    
    # Extract features
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        image_features = model.get_image_features(**image_inputs)
    
    print(f"Text features shape: {text_features.shape}")
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features norm: {torch.norm(text_features).item():.6f}")
    print(f"Image features norm: {torch.norm(image_features).item():.6f}")
    
    # Calculate similarity
    similarity = torch.cosine_similarity(text_features, image_features)
    print(f"Cosine similarity: {similarity.item():.6f}")
    
    return {
        'text_features': text_features,
        'image_features': image_features,
        'similarity': similarity.item(),
        'text_inputs': text_inputs,
        'image_inputs': image_inputs
    }

if __name__ == "__main__":
    # Test both configurations
    results_slow = test_clip_processor(use_fast=False)
    results_fast = test_clip_processor(use_fast=True)
    
    print(f"\n=== Comparison ===")
    print(f"Similarity difference: {abs(results_slow['similarity'] - results_fast['similarity']):.6f}")
    print(f"Text features difference (L2 norm): {torch.norm(results_slow['text_features'] - results_fast['text_features']).item():.6f}")
    print(f"Image features difference (L2 norm): {torch.norm(results_slow['image_features'] - results_fast['image_features']).item():.6f}")
    
    # Check if input processing is different
    text_ids_same = torch.equal(results_slow['text_inputs']['input_ids'], results_fast['text_inputs']['input_ids'])
    image_pixels_same = torch.allclose(results_slow['image_inputs']['pixel_values'], results_fast['image_inputs']['pixel_values'], atol=1e-6)
    
    print(f"Text input_ids identical: {text_ids_same}")
    print(f"Image pixel_values identical: {image_pixels_same}")
    
    if not text_ids_same:
        print("Text tokenization differs!")
        print(f"Slow tokenizer input_ids: {results_slow['text_inputs']['input_ids'][0]}")
        print(f"Fast tokenizer input_ids: {results_fast['text_inputs']['input_ids'][0]}")
    
    if not image_pixels_same:
        print("Image processing differs!")
        pixel_diff = torch.abs(results_slow['image_inputs']['pixel_values'] - results_fast['image_inputs']['pixel_values'])
        print(f"Max pixel difference: {pixel_diff.max().item():.6f}")