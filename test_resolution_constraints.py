#!/usr/bin/env python3
"""Test resolution constraints for various models"""

import sys
import json
from server import (
    AVAILABLE_MODELS, 
    validate_and_adjust_resolution,
    get_model_resolution_info,
    HUNYUAN_SUPPORTED_RESOLUTIONS
)

def test_model_resolution_constraints():
    """Test resolution constraints for each model."""
    
    print("Testing Resolution Constraints for All Models")
    print("=" * 80)
    
    # Test cases for various resolutions
    test_resolutions = [
        (512, 512),      # Small square
        (768, 768),      # Medium square
        (1024, 1024),    # Standard square
        (1536, 1024),    # Wide landscape
        (1920, 1080),    # 16:9
        (832, 1216),     # Portrait (Animagine recommended)
        (2048, 2048),    # Large square
        (4096, 4096),    # Very large (out of bounds)
    ]
    
    for model_key, config in AVAILABLE_MODELS.items():
        print(f"\n## {config.name} ({model_key})")
        print(f"   Description: {config.description}")
        print(f"   Range: {config.min_width}-{config.max_width} x {config.min_height}-{config.max_height}")
        
        if config.resolution_constraints:
            print(f"   Constraints: {config.resolution_constraints}")
        
        if config.recommended_resolutions:
            print(f"   Recommended: {config.recommended_resolutions}")
        
        # Get resolution info
        res_info = get_model_resolution_info(model_key)
        if "supported_resolutions" in res_info:
            print(f"   Supported (exact): {len(res_info['supported_resolutions'])} resolutions")
        
        print("\n   Resolution Tests:")
        for width, height in test_resolutions:
            adj_width, adj_height, warning = validate_and_adjust_resolution(model_key, width, height)
            
            if warning:
                print(f"   - {width}x{height} → {adj_width}x{adj_height} ⚠️  {warning}")
            elif (adj_width != width or adj_height != height):
                print(f"   - {width}x{height} → {adj_width}x{adj_height} (adjusted to fit bounds)")
            else:
                in_bounds = config.supports_dimensions(width, height)
                if in_bounds:
                    print(f"   - {width}x{height} ✓")
                else:
                    print(f"   - {width}x{height} → {adj_width}x{adj_height} (out of bounds)")

def test_hunyuan_specific():
    """Test HunyuanDiT specific resolution handling."""
    print("\n" + "=" * 80)
    print("HunyuanDiT Specific Resolution Tests")
    print("=" * 80)
    
    print(f"\nSupported resolutions ({len(HUNYUAN_SUPPORTED_RESOLUTIONS)} total):")
    for w, h in HUNYUAN_SUPPORTED_RESOLUTIONS:
        ratio = w / h
        print(f"  - {w}x{h} (ratio: {ratio:.2f})")
    
    # Test problematic resolutions
    print("\nProblematic resolution adjustments:")
    problematic = [
        (1536, 1024),  # User's failing case
        (1920, 1080),  # 16:9
        (800, 600),    # 4:3 but not supported
        (1600, 900),   # 16:9
    ]
    
    for width, height in problematic:
        adj_width, adj_height, warning = validate_and_adjust_resolution("hunyuan", width, height)
        print(f"\n  {width}x{height} (ratio: {width/height:.2f})")
        print(f"  → {adj_width}x{adj_height} (ratio: {adj_width/adj_height:.2f})")
        if warning:
            print(f"  ⚠️  {warning}")

def test_animagine_recommendations():
    """Test Animagine XL recommended resolutions."""
    print("\n" + "=" * 80)
    print("Animagine XL 4.0 Resolution Recommendations")
    print("=" * 80)
    
    res_info = get_model_resolution_info("anime")
    print(f"\nRecommended resolutions: {AVAILABLE_MODELS['anime'].recommended_resolutions}")
    print(f"Prompt format: {res_info.get('prompt_format', 'N/A')}")
    print(f"Quality tags: {res_info.get('quality_tags', [])}")
    
    # Test each recommended resolution
    print("\nValidating recommended resolutions:")
    for width, height in AVAILABLE_MODELS['anime'].recommended_resolutions:
        adj_width, adj_height, warning = validate_and_adjust_resolution("anime", width, height)
        status = "✓" if (adj_width == width and adj_height == height and not warning) else "✗"
        print(f"  - {width}x{height} {status}")

if __name__ == "__main__":
    test_model_resolution_constraints()
    test_hunyuan_specific()
    test_animagine_recommendations()
    
    print("\n" + "=" * 80)
    print("Resolution constraint tests complete!")