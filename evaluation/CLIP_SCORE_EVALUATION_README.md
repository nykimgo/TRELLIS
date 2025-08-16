# CLIP Score Evaluation for TRELLIS

This implementation provides CLIP Score evaluation for both original Toys4k dataset and TRELLIS-generated 3D assets.

## Overview

The CLIP Score evaluator measures the semantic alignment between 3D assets and their text descriptions by:

1. **3D Asset Rendering**: Renders each 3D asset from 8 viewpoints (45° intervals, pitch=30°, radius=2, FoV=40°)
2. **Feature Extraction**: Uses CLIP to extract features from rendered images and text captions
3. **Similarity Calculation**: Computes cosine similarity between image and text features
4. **Score Aggregation**: Averages across 8 views per asset, then across all assets

## Evaluators Available

### 1. Toys4k Dataset Evaluator
- `toys4k_clip_evaluator.py` - Evaluates original Toys4k dataset assets
- `clip_score_evaluator.py` - General-purpose CLIP evaluator

### 2. TRELLIS Generated Assets Evaluator  
- `trellis_generated_clip_evaluator.py` - **NEW** Evaluates TRELLIS-generated 3D assets
- `example_trellis_generated_evaluation.py` - Usage examples for generated assets
- `test_trellis_generated_evaluator.py` - Test suite for generated assets evaluator

### 3. Supporting Files
- `requirements_clip_eval.txt` - Required dependencies
- `test_basic_functionality.py` - Basic functionality tests (original)
- `example_clip_evaluation.py` - Usage examples (original)

## Dataset Structures

### 1. Original Toys4k Dataset
Located at `/mnt/nas/Benchmark_Datatset/Toys4k/`:
```
Toys4k/
├── metadata.csv                    # Asset metadata with captions
├── toys4k_obj_files/              # 3D assets in OBJ format
│   ├── airplane/airplane_000/mesh.obj
│   ├── boat/boat_001/mesh.obj
│   └── ...
└── toys4k_blend_files/            # 3D assets in Blender format
```

### 2. TRELLIS Generated Assets
Located at `/mnt/nas/tmp/nayeon/`:
```
/mnt/nas/tmp/nayeon/
├── sampled_data_100_random.csv                    # 100 sampled objects metadata
├── sampled_data_100_random_results_part01.xlsx   # LLM-augmented prompts
├── sampled_data_100_random_results_part02.xlsx   # (and other parts...)
└── outputs/TRELLIS-text-large/20250816/         # Generated 3D assets
    ├── gemma3_1b/
    │   ├── Giraffe/
    │   │   ├── Giraffe_TRELLIS-text-large_gemma3_1b_998646.glb
    │   │   └── Giraffe_TRELLIS-text-large_gemma3_1b_998646.ply
    │   └── Pig/...
    ├── qwen3_0.6b/...
    └── deepseek-r1_1.5b/...
```

## Installation

```bash
pip install -r requirements_clip_eval.txt
```

## Usage

### A. Original Toys4k Dataset Evaluation

#### Quick Test (5 assets)
```bash
python toys4k_clip_evaluator.py --max_assets 5 --output_path test_results.csv
```

#### Full Evaluation
```bash
python toys4k_clip_evaluator.py --output_path full_toys4k_clip_scores.csv
```

#### Custom CLIP Model
```bash
python toys4k_clip_evaluator.py --clip_model openai/clip-vit-large-patch14 --output_path results.csv
```

### B. TRELLIS Generated Assets Evaluation

#### Quick Test (5 generated assets)
```bash
python trellis_generated_clip_evaluator.py --max_assets 5 --save_path test_generated_results.csv
```

#### Evaluate Specific LLM Models
```bash
python trellis_generated_clip_evaluator.py --llm_models gemma3 qwen3 --save_path specific_models_results.csv
```

#### Full Evaluation of Generated Assets
```bash
python trellis_generated_clip_evaluator.py --save_path full_generated_results.csv
```

#### Run Example Evaluation
```bash
python example_trellis_generated_evaluation.py
```

## Command Line Arguments

### A. Original Toys4k Evaluator (`toys4k_clip_evaluator.py`)
- `--dataset_path`: Path to Toys4k dataset (default: `/mnt/nas/Benchmark_Datatset/Toys4k`)
- `--output_path`: Output CSV file path (default: `toys4k_clip_scores.csv`)
- `--clip_model`: CLIP model name (default: `openai/clip-vit-base-patch32`)
- `--max_assets`: Maximum assets to evaluate (for testing)

### B. TRELLIS Generated Evaluator (`trellis_generated_clip_evaluator.py`)
- `--results_excel`: Path to Excel file with LLM results (default: `/mnt/nas/tmp/nayeon/sampled_data_100_random_results_part01.xlsx`)
- `--output_base_path`: Base path to generated outputs (default: `/mnt/nas/tmp/nayeon/outputs/TRELLIS-text-large/20250816`)
- `--save_path`: Path to save results CSV file (default: `trellis_generated_clip_scores.csv`)
- `--clip_model`: CLIP model name (default: `openai/clip-vit-base-patch32`)
- `--max_assets`: Maximum assets to evaluate (for testing)
- `--llm_models`: List of LLM models to evaluate (e.g., `--llm_models gemma3 qwen3`)

## Output Files

### Detailed Results CSV
Contains per-asset evaluation results:
- `sha256`: Asset identifier
- `file_identifier`: Asset filename
- `aesthetic_score`: Original aesthetic score
- `clip_score`: Calculated CLIP score (-1 to 1)
- `num_views_rendered`: Number of successfully rendered views
- `success`: Whether evaluation succeeded
- `caption_used`: Text caption used for evaluation
- `error`: Error message if evaluation failed

### Summary JSON
Contains aggregated metrics:
- `mean_clip_score`: Average CLIP score across all assets
- `mean_clip_score_scaled`: CLIP score × 100 (paper convention)
- `total_assets`: Total number of assets
- `successful_evaluations`: Number of successful evaluations
- `success_rate`: Percentage of successful evaluations

## Implementation Details

### Rendering Configuration
- **Viewpoints**: 8 views at yaw angles [0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]
- **Camera**: Pitch=30°, Radius=2, FoV=40°
- **Resolution**: 512×512 pixels
- **Background**: White
- **Renderer**: matplotlib with 3D polygon collection

### CLIP Processing
- **Model**: Uses Hugging Face transformers CLIP implementation
- **Text**: Processes the most detailed caption for each asset
- **Images**: Processes all 8 rendered views
- **Features**: L2-normalized embeddings
- **Similarity**: Cosine similarity between image and text features

### Score Calculation
1. Extract CLIP features for 8 rendered views and 1 text caption
2. Calculate cosine similarity between each image and the text
3. Average similarities across all 8 views → asset CLIP score
4. Average asset scores across all assets → final CLIP score
5. Multiply by 100 for reporting (paper convention)

## Performance Considerations

- **Memory**: CLIP model requires ~1GB GPU memory
- **Speed**: ~30-60 seconds per asset (depending on mesh complexity)
- **Full Dataset**: 3229 assets × 1 minute ≈ 54 hours for complete evaluation
- **Recommendations**: 
  - Use GPU for CLIP inference
  - Start with `--max_assets` for testing
  - Run in batches for large-scale evaluation

## Error Handling

The evaluator handles various error conditions:
- Missing asset files
- Corrupted mesh data
- Rendering failures
- CLIP model errors

Failed evaluations are logged but don't stop the overall process.

## Example Results

```
=== Toys4k CLIP Score Evaluation Results ===
Dataset: /mnt/nas/Benchmark_Datatset/Toys4k
Total assets: 3229
Successful evaluations: 3150
Success rate: 97.55%
Mean CLIP Score: 0.2847
Mean CLIP Score (×100): 28.47
```

## Testing

Run basic functionality tests:
```bash
python test_basic_functionality.py
```

This verifies:
- Dataset structure and accessibility
- Metadata parsing
- 3D asset loading
- Rendering setup

## Troubleshooting

### CLIP Model Loading Issues
- Ensure torch version ≥ 2.6 for security compliance
- Use safetensors models when available
- Check GPU memory availability

### Rendering Issues
- Matplotlib backend conflicts: use `matplotlib.use('Agg')`
- Memory issues with large meshes: consider mesh simplification
- Display issues on headless servers: ensure proper backend configuration

### Dataset Issues
- Verify dataset path: `/mnt/nas/Benchmark_Datatset/Toys4k`
- Check file permissions
- Ensure both `metadata.csv` and `toys4k_obj_files/` exist

## Citation

If you use this CLIP Score evaluation in your research, please cite the TRELLIS paper and acknowledge the Toys4k dataset.