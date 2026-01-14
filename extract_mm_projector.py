#!/usr/bin/env python3
"""
Utility script to extract multimodal projector weights from LLaVA checkpoint
and save them in PyTorch 2.6+ compatible format (weights_only=True).

This allows using the fine-tuned mm_projector for image selection without
modifying the main model codebase.

Can read parameters from configuration file for consistency with the project's
config-driven approach.
"""

import torch
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {e}")


def get_extraction_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract mm_projector extraction configuration from the main config.
    
    Args:
        config: Main configuration dictionary
        
    Returns:
        Extraction configuration with defaults
    """
    # Default extraction configuration
    default_config = {
        "checkpoint_path": None,
        "output_path": "mm_projector_clean.bin",
        "auto_find_latest": False,
        "checkpoints_directory": "checkpoints",
        "allowed_extensions": [".bin", ".pth"],
        "verify_loading": True
    }
    
    # Get extraction config from the main config, or use defaults
    extraction_config = config.get("mm_projector_extraction", {})
    
    # Merge with defaults
    for key, default_value in default_config.items():
        if key not in extraction_config:
            extraction_config[key] = default_value
    
    # Auto-derive checkpoint path from model config if not specified
    if not extraction_config["checkpoint_path"] and "model" in config:
        model_config = config["model"]
        
        # Try different possible paths in order of preference
        possible_paths = [
            model_config.get("checkpoint_path"),
            model_config.get("pretrain_mm_mlp_adapter"),
            model_config.get("model_name_or_path")
        ]
        
        for path in possible_paths:
            if path and os.path.exists(path):
                extraction_config["checkpoint_path"] = path
                print(f"📋 Auto-derived checkpoint path from config: {path}")
                break
    
    return extraction_config


def find_checkpoint_file(checkpoint_path: str) -> str:
    """
    Find the actual checkpoint file from a given path (file or directory).
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
        
    Returns:
        Path to the actual checkpoint file
    """
    if os.path.isfile(checkpoint_path):
        return checkpoint_path
    
    if os.path.isdir(checkpoint_path):
        print(f"🔍 Checkpoint path is a directory, searching for checkpoint files...")
        
        # Look for common checkpoint file patterns
        # Order matters: check most likely locations first
        checkpoint_patterns = [
            "non_lora_trainables.bin",   # LLaVA non-LoRA weights (most likely for mm_projector)
            "pytorch_model.bin",         # Standard PyTorch model
            "model.safetensors",         # SafeTensors format
            "adapter_model.safetensors", # LoRA adapter weights (might contain mm_projector)
            "*.bin", "*.pth"            # Any other PyTorch files
        ]
        
        checkpoint_dir = Path(checkpoint_path)
        
        # First, look for numbered checkpoint subdirectories (e.g., checkpoint-6500)
        checkpoint_subdirs = [d for d in checkpoint_dir.iterdir() 
                             if d.is_dir() and d.name.startswith('checkpoint-')]
        
        if checkpoint_subdirs:
            # Use the latest checkpoint subdirectory
            latest_checkpoint_dir = max(checkpoint_subdirs, 
                                      key=lambda d: int(d.name.split('-')[-1]))
            print(f"🔍 Found checkpoint subdirectory: {latest_checkpoint_dir}")
            
            # Look for checkpoint files in the subdirectory
            for pattern in checkpoint_patterns:
                if '*' in pattern:
                    files = list(latest_checkpoint_dir.glob(pattern))
                else:
                    files = [latest_checkpoint_dir / pattern]
                    files = [f for f in files if f.exists()]
                
                if files:
                    checkpoint_file = str(files[0])
                    print(f"✅ Found checkpoint file: {checkpoint_file}")
                    return checkpoint_file
        
        # If no subdirectories, look directly in the main directory
        for pattern in checkpoint_patterns:
            if '*' in pattern:
                files = list(checkpoint_dir.glob(pattern))
            else:
                files = [checkpoint_dir / pattern]
                files = [f for f in files if f.exists()]
            
            if files:
                checkpoint_file = str(files[0])
                print(f"✅ Found checkpoint file: {checkpoint_file}")
                return checkpoint_file
        
        raise FileNotFoundError(f"No checkpoint files found in directory: {checkpoint_path}")
    
    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")


def extract_mm_projector_weights(checkpoint_path: str, output_path: str):
    """
    Extract mm_projector weights from a full model checkpoint and save them
    in a clean, PyTorch 2.6+ compatible format.
    
    Since the mm_projector was fine-tuned during LoRA training, we need to check
    multiple possible locations for the updated weights.
    
    Args:
        checkpoint_path: Path to the full model checkpoint (file or directory)
        output_path: Path where to save the clean mm_projector weights
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # If it's a directory, we'll check multiple files
    if os.path.isdir(checkpoint_path):
        return extract_from_checkpoint_directory(checkpoint_path, output_path)
    else:
        # Single file
        return extract_from_single_file(checkpoint_path, output_path)


def reconstruct_lora_weights(lora_weights: dict, base_weights_path: str, lora_config: Optional[dict] = None):
    """
    Reconstruct full weights from LoRA weights and base weights.
    
    Args:
        lora_weights: Dictionary containing lora_A and lora_B weights
        base_weights_path: Path to base mm_projector weights
        lora_config: LoRA configuration (r, alpha, etc.) or None
    
    Returns:
        Dictionary with reconstructed full weights
    """
    print(f"🔧 Reconstructing full weights from LoRA weights...")
    print(f"📁 Loading base weights from: {base_weights_path}")
    
    # Load base weights
    try:
        if not os.path.exists(base_weights_path):
            raise FileNotFoundError(f"Base weights not found: {base_weights_path}")
            
        base_checkpoint = load_checkpoint_file(base_weights_path)
        
        # Extract base mm_projector weights
        base_mm_projector = {}
        if isinstance(base_checkpoint, dict):
            for key, value in base_checkpoint.items():
                if 'mm_projector' in key:
                    clean_key = key
                    if 'mm_projector.' in clean_key:
                        # Keep only the layer name after mm_projector
                        clean_key = clean_key.split('mm_projector.')[-1]
                        if clean_key in ['weight', 'bias']:  # Standard layer names
                            base_mm_projector[clean_key] = value
        
        if not base_mm_projector:
            raise ValueError("No base mm_projector weights found in base checkpoint")
            
        print(f"✅ Loaded {len(base_mm_projector)} base mm_projector parameters")
        
    except Exception as e:
        print(f"❌ Error loading base weights: {e}")
        return None
    
    # Parse LoRA weights
    lora_A = None
    lora_B = None
    
    for key, value in lora_weights.items():
        if 'lora_A' in key:
            lora_A = value
            print(f"Found LoRA A: {key} -> shape {value.shape}")
        elif 'lora_B' in key:
            lora_B = value
            print(f"Found LoRA B: {key} -> shape {value.shape}")
    
    if lora_A is None or lora_B is None:
        print("❌ Missing LoRA A or B weights")
        return None
    
    # Get LoRA scaling factor (alpha / r)
    # Default values if not provided
    lora_r = lora_config.get('lora_r', 128) if lora_config else 128
    lora_alpha = lora_config.get('lora_alpha', 32) if lora_config else 32
    scaling = lora_alpha / lora_r
    
    print(f"📊 LoRA config: r={lora_r}, alpha={lora_alpha}, scaling={scaling}")
    
    # Reconstruct full weights: W = W_base + scaling * (B @ A)
    try:
        # Compute LoRA delta: scaling * (lora_B @ lora_A)
        lora_delta = scaling * torch.matmul(lora_B, lora_A)
        print(f"✅ Computed LoRA delta: shape {lora_delta.shape}")
        
        # Add to base weights
        reconstructed_weights = {}
        
        # Find matching base weight (usually 'weight')
        if 'weight' in base_mm_projector:
            base_weight = base_mm_projector['weight']
            
            if base_weight.shape == lora_delta.shape:
                reconstructed_weight = base_weight + lora_delta
                reconstructed_weights['mm_projector.weight'] = reconstructed_weight
                print(f"✅ Reconstructed mm_projector.weight: shape {reconstructed_weight.shape}")
            else:
                print(f"❌ Shape mismatch: base {base_weight.shape} vs LoRA delta {lora_delta.shape}")
                return None
        else:
            print("❌ No 'weight' found in base mm_projector")
            return None
        
        # Copy bias if it exists
        if 'bias' in base_mm_projector:
            reconstructed_weights['mm_projector.bias'] = base_mm_projector['bias']
            print(f"✅ Copied bias: shape {base_mm_projector['bias'].shape}")
        
        print(f"🎉 Successfully reconstructed {len(reconstructed_weights)} parameters")
        return reconstructed_weights
        
    except Exception as e:
        print(f"❌ Error reconstructing weights: {e}")
        return None


def extract_from_checkpoint_directory(checkpoint_dir: str, output_path: str):
    """
    Extract mm_projector from a checkpoint directory with multiple files.
    Checks all possible locations for fine-tuned mm_projector weights.
    """
    print(f"🔍 Searching checkpoint directory: {checkpoint_dir}")
    
    # Find the checkpoint subdirectory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_subdirs = [d for d in checkpoint_path.iterdir() 
                         if d.is_dir() and d.name.startswith('checkpoint-')]
    
    if checkpoint_subdirs:
        latest_checkpoint_dir = max(checkpoint_subdirs, 
                                  key=lambda d: int(d.name.split('-')[-1]))
        search_dir = latest_checkpoint_dir
        print(f"🔍 Using latest checkpoint: {search_dir}")
    else:
        search_dir = checkpoint_path
        print(f"🔍 Searching in main directory: {search_dir}")
    
    # Check multiple files in order of likelihood
    search_files = [
        ("non_lora_trainables.bin", "Non-LoRA trainable weights"),
        ("adapter_model.safetensors", "LoRA adapter weights"),
        ("pytorch_model.bin", "PyTorch model"),
        ("model.safetensors", "SafeTensors model")
    ]
    
    mm_projector_weights = {}
    found_in_file = None
    is_lora_weights = False
    
    for filename, description in search_files:
        file_path = search_dir / filename
        if file_path.exists():
            print(f"📁 Checking {description}: {file_path}")
            try:
                weights = load_checkpoint_file(str(file_path))
                projector_weights = extract_mm_projector_from_weights(weights, str(file_path))
                
                if projector_weights:
                    # Check if these are LoRA weights
                    has_lora_a = any('lora_A' in key for key in projector_weights.keys())
                    has_lora_b = any('lora_B' in key for key in projector_weights.keys())
                    
                    if has_lora_a and has_lora_b:
                        print(f"🔍 Detected LoRA weights in {filename}")
                        is_lora_weights = True
                        mm_projector_weights.update(projector_weights)
                        found_in_file = str(file_path)
                        break
                    else:
                        print(f"✅ Found {len(projector_weights)} full mm_projector parameters in {filename}")
                        mm_projector_weights.update(projector_weights)
                        found_in_file = str(file_path)
                        break
                else:
                    print(f"❌ No mm_projector weights found in {filename}")
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
        else:
            print(f"⚪ File not found: {filename}")
    
    if not mm_projector_weights:
        print("❌ No mm_projector weights found in any checkpoint files")
        print("💡 This might mean:")
        print("   - The mm_projector wasn't actually fine-tuned")
        print("   - The weights are stored in a different format/location")
        print("   - You might need to use the original pretrained mm_projector")
        return False
    
    # Handle LoRA weights - reconstruct full weights
    if is_lora_weights:
        print("\n🔧 Found LoRA weights - need to reconstruct full mm_projector...")
        
        # Load config to get LoRA parameters and base model path
        try:
            config_path = "configs/image_text_config.json"
            if os.path.exists(config_path):
                config = load_config(config_path)
                base_weights_path = config.get("model", {}).get("pretrain_mm_mlp_adapter")
                lora_config = {
                    "lora_r": config.get("training", {}).get("lora_r", 128),
                    "lora_alpha": config.get("training", {}).get("lora_alpha", 32)
                }
            else:
                print("⚠️  Config file not found, using default LoRA parameters")
                base_weights_path = "checkpoints/llava-vicuna-v1-5-7b-stage1/mm_projector.bin"
                lora_config = {"lora_r": 128, "lora_alpha": 32}
            
            if not base_weights_path:
                print("❌ No base mm_projector path found in config")
                return False
            
            # Reconstruct full weights
            full_weights = reconstruct_lora_weights(mm_projector_weights, base_weights_path, lora_config)
            
            if full_weights:
                mm_projector_weights = full_weights
                print("✅ Successfully reconstructed full mm_projector weights from LoRA")
            else:
                print("❌ Failed to reconstruct weights from LoRA")
                return False
                
        except Exception as e:
            print(f"❌ Error during LoRA reconstruction: {e}")
            return False
    
    return save_clean_weights(mm_projector_weights, output_path, found_in_file or "unknown")


def extract_from_single_file(checkpoint_path: str, output_path: str):
    """Extract mm_projector from a single checkpoint file."""
    print(f"📁 Loading single checkpoint file: {checkpoint_path}")
    
    try:
        weights = load_checkpoint_file(checkpoint_path)
        mm_projector_weights = extract_mm_projector_from_weights(weights, checkpoint_path)
        
        if not mm_projector_weights:
            print("❌ No mm_projector weights found in checkpoint")
            return False
            
        return save_clean_weights(mm_projector_weights, output_path, checkpoint_path)
        
    except Exception as e:
        print(f"❌ Error processing checkpoint: {e}")
        return False


def load_checkpoint_file(file_path: str):
    """Load a checkpoint file (PyTorch or SafeTensors format)."""
    if file_path.endswith('.safetensors'):
        # Load SafeTensors format
        try:
            from safetensors.torch import load_file
            checkpoint = load_file(file_path)
            print("✅ Successfully loaded SafeTensors checkpoint")
            return checkpoint
        except ImportError:
            raise ImportError("SafeTensors library not found. Install with: pip install safetensors")
    else:
        # Load PyTorch format
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
        print("✅ Successfully loaded PyTorch checkpoint")
        return checkpoint


def extract_mm_projector_from_weights(checkpoint, source_file: str):
    """Extract mm_projector weights from loaded checkpoint."""
    mm_projector_weights = {}
    
    # Look for mm_projector weights in the checkpoint
    if isinstance(checkpoint, dict):
        # Check different possible keys where mm_projector might be stored
        possible_keys = ['state_dict', 'model', 'model_state_dict']
        weights_dict = checkpoint
        
        for key in possible_keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                weights_dict = checkpoint[key]
                print(f"Found weights in checkpoint['{key}']")
                break
        
        # Ensure weights_dict is a dictionary
        if isinstance(weights_dict, dict):
            # Extract mm_projector weights
            for key, value in weights_dict.items():
                if 'mm_projector' in key:
                    # Clean up the key name
                    clean_key = key
                    if 'base_model.' in clean_key:
                        clean_key = clean_key.replace('base_model.', '')
                    if 'model.' in clean_key and clean_key.startswith('model.'):
                        clean_key = clean_key.replace('model.', '', 1)
                    
                    mm_projector_weights[clean_key] = value
                    print(f"Found: {key} -> {clean_key}")
        else:
            print(f"⚠️  Warning: weights_dict is not a dictionary, it's a {type(weights_dict)}")
    
    if not mm_projector_weights:
        print(f"❌ No mm_projector weights found in {source_file}")
        print("Available keys:")
        if isinstance(checkpoint, dict):
            weights_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
            if isinstance(weights_dict, dict):
                for key in sorted(weights_dict.keys())[:20]:  # Show first 20 keys
                    print(f"  {key}")
                if len(weights_dict) > 20:
                    print(f"  ... and {len(weights_dict) - 20} more")
            else:
                # If not nested, show checkpoint keys directly
                for key in sorted(checkpoint.keys())[:20]:
                    print(f"  {key}")
                if len(checkpoint) > 20:
                    print(f"  ... and {len(checkpoint) - 20} more")
    
    return mm_projector_weights


def save_clean_weights(mm_projector_weights: dict, output_path: str, source_file: str):
    """Save mm_projector weights in PyTorch 2.6+ compatible format."""
    print(f"✅ Extracted {len(mm_projector_weights)} mm_projector parameters from {source_file}")
    
    # Save in clean format that's compatible with weights_only=True
    print(f"Saving clean mm_projector weights to: {output_path}")
    
    # Create directory only if output_path contains a directory component
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there's actually a directory to create
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        torch.save(mm_projector_weights, output_path)
        print("✅ Successfully saved clean mm_projector weights")
        
        # Verify the saved file can be loaded with weights_only=True
        print("🔍 Verifying saved file...")
        test_load = torch.load(output_path, map_location='cpu', weights_only=True)
        print(f"✅ Verification successful! File can be loaded with weights_only=True")
        print(f"   Contains {len(test_load)} parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving weights: {e}")
        return False


def find_latest_checkpoint(checkpoints_dir: str, allowed_extensions: list) -> str:
    """
    Find the latest checkpoint in the specified directory.
    
    Args:
        checkpoints_dir: Directory to search for checkpoints
        allowed_extensions: List of allowed file extensions
        
    Returns:
        Path to the latest checkpoint
    """
    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Find all checkpoint files with allowed extensions
    checkpoint_files = []
    for ext in allowed_extensions:
        checkpoint_files.extend(list(checkpoints_path.rglob(f"*{ext}")))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoints_dir} with extensions {allowed_extensions}")
    
    # Use the most recent one
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return str(latest_checkpoint)


def main():
    parser = argparse.ArgumentParser(description="Extract mm_projector weights from LLaVA checkpoint")
    parser.add_argument("--config", "-cfg", default="configs/image_text_config.json",
                       help="Path to configuration file (default: configs/image_text_config.json)")
    parser.add_argument("--checkpoint", "-c",
                       help="Path to the full model checkpoint (overrides config)")
    parser.add_argument("--output", "-o",
                       help="Output path for clean mm_projector weights (overrides config)")
    parser.add_argument("--auto-find", "-a", action="store_true",
                       help="Auto-find the latest checkpoint (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if os.path.exists(args.config):
        try:
            config = load_config(args.config)
            print(f"📋 Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load config file {args.config}: {e}")
            print("Using command line arguments only...")
    else:
        print(f"⚠️  Config file not found: {args.config}")
        print("Using command line arguments only...")
    
    # Get extraction configuration with defaults
    extraction_config = get_extraction_config(config)
    
    # Command line arguments override config file
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"📋 Using checkpoint from command line: {checkpoint_path}")
    elif args.auto_find or extraction_config.get("auto_find_latest", False):
        try:
            checkpoints_dir = extraction_config.get("checkpoints_directory", "checkpoints")
            allowed_extensions = extraction_config.get("allowed_extensions", [".bin", ".pth"])
            checkpoint_path = find_latest_checkpoint(checkpoints_dir, allowed_extensions)
            print(f"🔍 Auto-found latest checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"❌ Error finding latest checkpoint: {e}")
            return
    elif extraction_config.get("checkpoint_path"):
        checkpoint_path = extraction_config["checkpoint_path"]
        print(f"📋 Using checkpoint from config: {checkpoint_path}")
    else:
        print("❌ No checkpoint path specified. Use --checkpoint, --auto-find, or configure in config file.")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint file not found: {checkpoint_path}")
        return
    
    # Set output path (CLI overrides config)
    if args.output:
        output_path = args.output
        print(f"📋 Using output path from command line: {output_path}")
    else:
        output_path = extraction_config.get("output_path", "mm_projector_clean.bin")
        print(f"📋 Using output path from config: {output_path}")
    
    # Extract and save
    print(f"\n🚀 Starting mm_projector extraction...")
    success = extract_mm_projector_weights(checkpoint_path, output_path)
    
    if success:
        print(f"\n🎉 Success! Clean mm_projector weights saved to: {output_path}")
        print(f"\nYou can now use this file in your image selection code with:")
        print(f"  torch.load('{output_path}', weights_only=True)")
        
        # Show next steps
        print(f"\n📝 Next steps:")
        print(f"1. Update your image selection config to use: {output_path}")
        print(f"2. Your main model code remains unchanged")
        print(f"3. The image selection will use PyTorch 2.6+ compatible weights")
    else:
        print(f"\n❌ Failed to extract mm_projector weights")


if __name__ == "__main__":
    main() 