#!/usr/bin/env python3

import torch
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_bwd.*")
warnings.filterwarnings("ignore", message=".*torch.library.impl_abstract.*")

try:
    from trellis import TrellisImageTo3DPipeline
    
    print("Loading TRELLIS pipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    
    print("Pipeline models:")
    if hasattr(pipeline, 'models'):
        for key, value in pipeline.models.items():
            print(f"  {key}: {type(value)} - {'None' if value is None else 'OK'}")
    else:
        print("  No models attribute")
        
    print(f"\nPipeline attributes: {[attr for attr in dir(pipeline) if not attr.startswith('_')]}")
    
    # Test run method
    print(f"\nHas run method: {hasattr(pipeline, 'run')}")
    if hasattr(pipeline, 'run'):
        print(f"Run method signature: {pipeline.run.__doc__}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()