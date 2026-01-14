import os
import sys

# Add the project root to the path before any other imports
# This ensures the src module can be found
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, project_root)

import argparse
import torch
import json
import pandas as pd
from src.constants import IMAGE_TOKEN_INDEX
from src.conversation import conv_templates, SeparatorStyle
from src.model.builder import load_pretrained_model
from src.utils import disable_torch_init
from src.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from src.config_utils import load_config
import lmdb
import pickle
import numpy as np
import time
from PIL import Image
from typing import Dict, Any, List, Optional

from src.mm_utils import get_model_name_from_path
from src.constants import DEFAULT_IMAGE_TOKEN
from src.prompt_manager import PromptManager
from src.utils import get_attention_implementation, print_attention_info

# Default configuration path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "configs", "image_text_config.json")
# Load model version for template selection
_config = load_config(DEFAULT_CONFIG_PATH)
TEMPLATE_VERSION = _config["model"].get("version", "v1")
print(f"Inference using conversation template version: '{TEMPLATE_VERSION}'")
if TEMPLATE_VERSION in conv_templates:
    print(f"Found template '{TEMPLATE_VERSION}' in available templates")
else:
    print(f"Warning: Template version '{TEMPLATE_VERSION}' not found, will fallback to 'v1'")

def open_lmdb(lmdb_path):
    return lmdb.open(
        lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
    )

def normalize_path(path):
    # Replace backslashes with forward slashes
    normalized = path.replace('\\', '/')
    # Remove any double slashes
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
    return normalized

def get_npy_feature_from_lmdb_with_jpg(env, key):
    key = normalize_path(key)
    # Look up in LMDB
    with env.begin() as txn:
        value = txn.get(key.encode())
        if value is None:
            print(f'Warning: "Key \'{key}\' not found in LMDB."')
            raise KeyError(f"Key '{key}' not found in LMDB.")
        feature = pickle.loads(value)
    return torch.tensor(feature, dtype=torch.float32)

def load_image_features(lmdb_env, image_paths, max_images, image_feature_dim):
    """
    Load image features from LMDB, pad/truncate to max_images, and create a mask.
    """
    # Preallocate tensors for features and masks
    images_features = torch.zeros(
        (max_images, image_feature_dim), dtype=torch.float32
    )
    images_mask = torch.zeros(max_images, dtype=torch.int16)
    img_paths = []

    for i, path in enumerate(image_paths):
        if i >= max_images:
            break
        path = os.path.normpath(path).replace("\\", "/")
        try:
            feature = get_npy_feature_from_lmdb_with_jpg(
                lmdb_env, path
            )
            images_features[i] = feature
            images_mask[i] = 1
            img_paths.append(path)
        except KeyError as e:
            print(f"Warning: {e}")
            continue

    return images_features, images_mask, img_paths

def inference(model, images_features, query, tokenizer, max_new_tokens=512, temperature=0.7, 
             do_sample=True, num_beams=3, top_p=0.9, top_k=50, repetition_penalty=1.0,
             min_new_tokens=1, length_penalty=1.0, no_repeat_ngram_size=3, early_stopping=True):
    conv = conv_templates.get(TEMPLATE_VERSION, conv_templates["v1"]).copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Calculate max context length more accurately
    model_max_length = getattr(tokenizer, 'model_max_length', 4096)
    # Conservative safety margin for generation and special tokens
    safety_margin = max_new_tokens + 100  # 100 tokens for special tokens, conversation template, etc.
    max_context_length = model_max_length - safety_margin
    
    # Tokenize and check length - use proper tokenization
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    
    # Handle truncation if needed
    if input_ids.shape[0] > max_context_length:
        print(f"Warning: Input too long ({input_ids.shape[0]} tokens > {max_context_length} max). Truncating...")
        
        # Truncate input_ids directly and decode back to get truncated prompt
        truncated_input_ids = input_ids[:max_context_length]
        
        # Convert back to text (this preserves tokenizer consistency)
        truncated_text = tokenizer.decode(truncated_input_ids, skip_special_tokens=True)
        
        # Rebuild conversation with truncated content
        conv = conv_templates.get(TEMPLATE_VERSION, conv_templates["v1"]).copy()
        conv.append_message(conv.roles[0], f"(Content truncated due to length) {truncated_text}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Re-tokenize the truncated prompt
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    
    # Ensure input_ids is properly shaped and on CUDA
    input_ids = input_ids.unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        try:
            outputs = model.generate(
                input_ids,
                images=images_features.unsqueeze(0).cuda(),
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_beams=num_beams,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        except RuntimeError as e:
            if "probability tensor" in str(e) or "CUDA" in str(e):
                print("Encountered error. Retrying with safer parameters.")
                outputs = model.generate(
                    input_ids,
                    images=images_features.unsqueeze(0).cuda(),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            else:
                raise

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != outputs[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    decoded_outputs = tokenizer.batch_decode(outputs[:, input_token_len:], skip_special_tokens=True)[0]
    decoded_outputs = decoded_outputs.strip()
    if decoded_outputs.endswith(stop_str):
        decoded_outputs = decoded_outputs[:-len(stop_str)]
    decoded_outputs = decoded_outputs.strip()
    
    # Clean up common prefix patterns that shouldn't be in summaries
    if decoded_outputs.lower().startswith("this article"):
        decoded_outputs = decoded_outputs[12:].strip()
    elif decoded_outputs.lower().startswith("the article"):
        decoded_outputs = decoded_outputs[11:].strip()
    
    return decoded_outputs

def inference_on_dataset(model, tokenizer, config):
    data_config = config["data"]
    inference_config = config["inference"]
    
    # Load the dataset from test file if available, otherwise fall back to data_path
    data_path = data_config.get("test_path", data_config["data_path"])
    print(f"Loading dataset from: {data_path}")
    
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Initialize LMDB environment using test LMDB if available
    lmdb_path = data_config.get("test_lmdb_path", data_config["lmdb_path"])
    print(f"Using LMDB from: {lmdb_path}")
    lmdb_env = open_lmdb(lmdb_path)
    
    # Apply sampling if test_subset_size is specified
    test_subset_size = data_config.get("test_subset_size", 0)
    if test_subset_size > 0 and len(df) > test_subset_size:
        print(f"Sampling {test_subset_size} examples from {len(df)} total examples")
        df = df.sample(test_subset_size, random_state=42)
    
    # Process each entry in the dataset
    results = []
    max_samples = inference_config.get("max_samples", 0)
    save_every_n_samples = inference_config.get("save_every_n_samples", 0)  # How often to save intermediate results
    
    # Create a function to save intermediate results
    def save_intermediate_results(results_data, output_path, beam_size, completed=False):
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # If there are no results yet, don't save
        if not results_data:
            return
            
        # Convert results to DataFrame and save
        results_df = pd.DataFrame(results_data)
        
        # If this is an intermediate save, add beam size and partial suffix to the filename
        if not completed:
            # Split the path to insert beam size and '_partial' before the extension
            base, ext = os.path.splitext(output_path)
            # Remove any existing beam size info from base filename to avoid duplication
            if '_beam_' in base:
                base = base.split('_beam_')[0]
            partial_path = f"{base}_beam_{beam_size}_partial{ext}"
            results_df.to_csv(partial_path, index=False)
            print(f"Intermediate results saved to {partial_path} ({len(results_data)} samples, beam_size={beam_size})")
        else:
            # Final save - check if beam size is already in the filename
            base, ext = os.path.splitext(output_path)
            if '_beam_' not in base:
                # Add beam size to the final filename if not already present
                final_path = f"{base}_beam_{beam_size}{ext}"
            else:
                final_path = output_path
            results_df.to_csv(final_path, index=False)
            print(f"Final results saved to {final_path} (CSV, beam_size={beam_size})")
    
    # Limit the number of samples if specified
    if max_samples > 0:
        df = df.head(max_samples)
    
    # Get output path from config
    csv_output_path = inference_config.get("csv_output_path", "results/inference_results.csv")
    
    # Get beam size for filename inclusion
    num_beams = inference_config.get("num_beams", 3)
    
    print(f"Running inference on {len(df)} samples...")
    
    for idx, row in df.iterrows():
        # Extract image paths - handle different possible column names
        image_paths = []
        if 'Image_Paths' in row:
            image_paths = row['Image_Paths']
        elif 'image_paths' in row:
            image_paths = row['image_paths']
            
        # Parse image paths if they're in string format (JSON)
        if isinstance(image_paths, str):
            try:
                image_paths = json.loads(image_paths)
            except:
                # If not valid JSON, treat as a single path
                image_paths = [image_paths]
        
        # Handle empty image paths
        if not image_paths:
            print(f"Warning: No image paths found for item {idx}")
            image_paths = []
              # Extract text content - adapt to your CSV column names
        text_content = None
        for column in ['Body', 'body', 'text', 'content', 'Text', 'Content']:
            if column in row and pd.notna(row[column]):
                text_content = str(row[column])
                break
                
        if text_content is None:
            raise ValueError(f"No text content found for item {idx}. Required text field missing.")
            
        # Handle text length more intelligently with tokenizer-aware truncation
        # First get the prompt template to understand overhead
        from src.prompt_manager import format_prompt_with_images
        temp_prompt = format_prompt_with_images("", len(image_paths), mode="inference")
        
        # Estimate template overhead in tokens (conservative estimate)
        template_tokens = len(tokenizer.encode(temp_prompt))
        
        # Calculate available space for text content
        model_max_length = getattr(tokenizer, 'model_max_length', 4096)
        max_new_tokens = inference_config.get("max_new_tokens", 256)
        safety_margin = 150  # For conversation template, special tokens, etc.
        
        # Use max_input_len from config as token limit (not characters!)
        max_input_len = data_config.get("max_input_len", 1536)
        
        # Use the more restrictive of: config limit or model capacity
        available_tokens_for_text = min(
            max_input_len,
            model_max_length - max_new_tokens - template_tokens - safety_margin
        )
        
        # Tokenize the text content to check if it fits
        text_tokens = tokenizer.encode(text_content, add_special_tokens=False)
        
        if len(text_tokens) > available_tokens_for_text:
            print(f"Warning: Text content for item {idx} too long ({len(text_tokens)} tokens > {available_tokens_for_text} available). Truncating.")
            # Truncate at token level, not character level
            truncated_tokens = text_tokens[:available_tokens_for_text]
            text_content = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
        # Load image features
        try:
            images_features, images_mask, img_paths = load_image_features(
                lmdb_env=lmdb_env,
                image_paths=image_paths,
                max_images=data_config["max_images"],
                image_feature_dim=data_config["image_feature_dim"],
            )
            
            has_images = len(img_paths) > 0
        except Exception as e:
            print(f"Error loading images for item {idx}: {str(e)}")
            # Don't fallback to empty tensors, propagate the error
            raise Exception(f"Failed to load images for item {idx}: {str(e)}")
        
        # Create prompt based on whether we have images using centralized prompt manager
        from src.prompt_manager import format_prompt_with_images
        prompt = format_prompt_with_images(text_content, len(img_paths), mode="inference")
        
        # Generate summary
        print(f"Processing item {idx+1}/{len(df)}...")
        try:
            summary = inference(
                model, 
                images_features.to(torch.float16).cuda() if torch.cuda.is_available() else images_features, 
                prompt, 
                tokenizer,
                max_new_tokens=inference_config.get("max_new_tokens", 512),
                temperature=inference_config.get("temperature", 0.7),
                do_sample=inference_config.get("do_sample", True),
                num_beams=inference_config.get("num_beams", 3),
                top_p=inference_config.get("top_p", 0.9),
                top_k=inference_config.get("top_k", 50),
                repetition_penalty=inference_config.get("repetition_penalty", 1.0),
                min_new_tokens=inference_config.get("min_new_tokens", 1),
                length_penalty=inference_config.get("length_penalty", 1.0),
                no_repeat_ngram_size=inference_config.get("no_repeat_ngram_size", 3),
                early_stopping=inference_config.get("early_stopping", True)
            )
        except Exception as e:
            print(f"Error generating summary for item {idx}: {str(e)}")
            summary = "Error generating summary."
        
        # Store results
        result = {
            'id': row.get('id', row.get('Article_ID', idx)),
            'text': text_content[:500] + "..." if len(text_content) > 500 else text_content,
            'image_paths': img_paths,
            'generated_summary': summary,
        }
        
        # Include reference summary if available in the dataset
        for summary_col in ['Summary', 'summary', 'reference', 'Reference']:
            if summary_col in row and pd.notna(row[summary_col]):
                result['reference_summary'] = row[summary_col]
                break
            
        results.append(result)
        
        # Save intermediate results if configured
        if save_every_n_samples > 0 and (idx + 1) % save_every_n_samples == 0:
            save_intermediate_results(results, csv_output_path, num_beams, completed=False)
        
        # Only print processing indicator without the summary
        print("-" * 80)
    
    # Save final results (all samples)
    save_intermediate_results(results, csv_output_path, num_beams, completed=True)

def run_inference(config_path):
    # Load configuration from file
    config = load_config(config_path)
    
    # Validate configuration before starting
    validate_config(config)
    
    disable_torch_init()
    
    # Prepare model arguments
    model_config = config["model"]
    
    # Create an args class that mimics the argparse Namespace
    class ModelArgs:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = ModelArgs(
        model_base=model_config["model_name_or_path"],
        pretrain_mm_mlp_adapter=model_config["pretrain_mm_mlp_adapter"]
    )
    
    # Load model
    tokenizer, model, context_len = load_pretrained_model(
        args=args, 
        stage2=model_config.get("checkpoint_path")
    )
    model = model.cuda()
    model.to(torch.float16)
    
    # Run inference
    inference_on_dataset(model, tokenizer, config)

def validate_config(config):
    """
    Validate configuration for potential truncation and token limit issues.
    """
    print("Validating configuration...")
    
    data_config = config["data"]
    inference_config = config["inference"]
    
    # Get token limits (max_input_len represents tokens, not characters)
    max_input_len = data_config.get("max_input_len", 1536)  # tokens
    max_new_tokens = inference_config.get("max_new_tokens", 256)
    
    # Estimate template overhead (conservative)
    estimated_template_overhead = 200  # tokens for prompts, conversation template, etc.
    
    # Check if limits are reasonable
    total_estimated_tokens = max_input_len + max_new_tokens + estimated_template_overhead
    
    print(f"Configuration summary:")
    print(f"  - Max input tokens (max_input_len): {max_input_len}")
    print(f"  - Max new tokens: {max_new_tokens}")
    print(f"  - Estimated template overhead: {estimated_template_overhead}")
    print(f"  - Total estimated tokens: {total_estimated_tokens}")
    
    # Warn about potential issues
    if total_estimated_tokens > 3500:
        print(f"Warning: Total estimated tokens ({total_estimated_tokens}) may exceed some model limits")
    
    if max_new_tokens > 512:
        print(f"Warning: max_new_tokens ({max_new_tokens}) is quite high and may cause memory issues")
    
    # Check beam search settings
    num_beams = inference_config.get("num_beams", 1)
    if num_beams > 5:
        print(f"Warning: num_beams ({num_beams}) is high and will increase memory usage significantly")
    
    print("Configuration validation complete.\n")

def get_generation_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get generation parameters from config with fallbacks."""
    generation_config = config.get("generation", {})
    
    # Default parameters optimized for abstractive summarization
    default_params = {
        "max_new_tokens": 512,
        "temperature": 0.3,
        "do_sample": True,
        "num_beams": 2,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "early_stopping": True
    }
    
    # Override with config values
    params = default_params.copy()
    params.update(generation_config)
    
    return params

def infer_image_text_summarization(config_path: str, input_data_path: str, output_path: str, 
                                 model_path: Optional[str] = None, adapter_path: Optional[str] = None,
                                 max_samples: Optional[int] = None):
    """
    Perform inference using the image-text summarization model.
    
    Args:
        config_path: Path to configuration file
        input_data_path: Path to input JSONL file with image and text data
        output_path: Path to save inference results
        model_path: Optional override for model path
        adapter_path: Optional path to LoRA adapter
        max_samples: Maximum number of samples to process (for testing)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Get prompt manager and load prompts
    prompts_config_path = config.get("prompts_config_path", "configs/prompts_config.json")
    prompt_manager = PromptManager(prompts_config_path)
    
    # Detect and use Flash Attention if available
    attn_implementation = get_attention_implementation(prefer_flash_attention=True)
    print(f"Using attention implementation: {attn_implementation}")
    print_attention_info(attn_implementation)
    
    # Model configuration
    model_config = config["model"]
    model_name = model_path or model_config["model_name_or_path"]
    
    # Disable torch init for faster loading
    disable_torch_init()
    
    # Load model with Flash Attention support
    print(f"Loading model from: {model_name}")
    load_kwargs = {}
    if attn_implementation and attn_implementation != "eager":
        load_kwargs["attn_implementation"] = attn_implementation
        print(f"Loading model with {attn_implementation} attention")
    
    # Create args object that matches the expected signature
    class ModelArgs:
        def __init__(self, model_base, pretrain_mm_mlp_adapter=None):
            self.model_base = model_base
            self.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter

    args = ModelArgs(
        model_base=model_name,
        pretrain_mm_mlp_adapter=model_config.get("pretrain_mm_mlp_adapter")
    )
    
    tokenizer, model, context_len = load_pretrained_model(
        args=args,
        stage2=adapter_path or model_config.get("checkpoint_path")
    )

    # Run inference
    inference_on_dataset(model, tokenizer, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for image-text summarization using a config file")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the configuration file")
    args = parser.parse_args()
    
    run_inference(args.config)