import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

def load_config(config_path):
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def merge_configs(base_config, override_config=None):
    """Merge a base configuration with an override configuration."""
    if override_config is None:
        return base_config
    
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def config_to_args(config, arg_class=None):
    """Convert a config dict to an argument object that mimics argparse Namespace."""
    if arg_class is not None:
        # Create an instance of the specified dataclass
        args = arg_class()
        
        # Update fields from config
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    else:
        # Create a simple namespace object
        class Args:
            pass
        
        args = Args()
        
        # Add all config items as attributes
        for key, value in config.items():
            setattr(args, key, value)
    
    return args