# CLIP Score Evaluation for TRELLIS

This implementation provides CLIP Score evaluation for TRELLIS 3D generation results using the Toys4k dataset.

## Overview

The CLIP Score evaluator measures the semantic alignment between generated 3D assets and their text descriptions by:

1. **3D Asset Rendering**: Renders each 3D asset from 8 viewpoints (45° intervals, pitch=30°, radius=2, FoV=40°)
2. **Feature Extraction**: Uses CLIP to extract features from rendered images and text captions
3. **Similarity Calculation**: Computes cosine similarity between image and text features
4. **Score Aggregation**: Averages across 8 views per asset, then across all assets

## Files Created

- `toys4k_clip_evaluator.py` - Main evaluator for Toys4k dataset
- `clip_score_evaluator.py` - General-purpose CLIP evaluator
- `requirements_clip_eval.txt` - Required dependencies
- `test_basic_functionality.py` - Basic functionality tests
- `example_clip_evaluation.py` - Usage examples

## Dataset Structure

The evaluator works with the Toys4k dataset at `/mnt/nas/Benchmark_Datatset/Toys4k/`:

```
Toys4k/
├── metadata.csv                    # Asset metadata with captions
├── toys4k_obj_files/              # 3D assets in OBJ format
│   ├── airplane/airplane_000/mesh.obj
│   ├── boat/boat_001/mesh.obj
│   └── ...
└── toys4k_blend_files/            # 3D assets in Blender format
```

## Installation

```bash
pip install -r requirements_clip_eval.txt
```

## Usage

### Quick Test (5 assets)
```bash
python toys4k_clip_evaluator.py --max_assets 5 --output_path test_results.csv
```

### Full Evaluation
```bash
python toys4k_clip_evaluator.py --output_path full_toys4k_clip_scores.csv
```

### Custom CLIP Model
```bash
python toys4k_clip_evaluator.py --clip_model openai/clip-vit-large-patch14 --output_path results.csv
```

## Command Line Arguments

- `--dataset_path`: Path to Toys4k dataset (default: `/mnt/nas/Benchmark_Datatset/Toys4k`)
- `--output_path`: Output CSV file path (default: `toys4k_clip_scores.csv`)
- `--clip_model`: CLIP model name (default: `openai/clip-vit-base-patch32`)
- `--max_assets`: Maximum assets to evaluate (for testing)

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