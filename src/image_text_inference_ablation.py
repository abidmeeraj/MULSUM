import os
import sys
import argparse
import torch
import json
import pandas as pd
import random
import numpy as np
from src.constants import IMAGE_TOKEN_INDEX
from src.conversation import conv_templates, SeparatorStyle
from src.model.builder import load_pretrained_model
from src.utils import disable_torch_init, get_attention_implementation, print_attention_info
from src.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from src.config_utils import load_config, config_to_args
from transformers import TextStreamer
import lmdb
import pickle

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

def get_all_lmdb_keys(lmdb_env):
    """Get all keys from the LMDB database"""
    keys = []
    with lmdb_env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            keys.append(key.decode())
    return keys

def load_random_image_features(lmdb_env, all_keys, max_images, image_feature_dim, num_images_to_select=None):
    """
    Load random image features from LMDB instead of using the original image paths.
    
    Args:
        lmdb_env: LMDB environment
        all_keys: List of all available keys in LMDB
        max_images: Maximum number of images to load
        image_feature_dim: Dimension of image features
        num_images_to_select: Number of random images to select (if None, uses max_images)
    
    Returns:
        images_features: Tensor of image features
        images_mask: Mask indicating which positions have valid images
        selected_keys: List of randomly selected keys
    """
    if num_images_to_select is None:
        num_images_to_select = max_images
    
    # Ensure we don't try to select more images than available
    num_images_to_select = min(num_images_to_select, len(all_keys), max_images)
    
    # Preallocate tensors for features and masks
    images_features = torch.zeros(
        (max_images, image_feature_dim), dtype=torch.float32
    )
    images_mask = torch.zeros(max_images, dtype=torch.int16)
    
    # Randomly select keys
    selected_keys = random.sample(all_keys, num_images_to_select)
    
    for i, key in enumerate(selected_keys):
        try:
            feature = get_npy_feature_from_lmdb_with_jpg(lmdb_env, key)
            images_features[i] = feature
            images_mask[i] = 1
        except KeyError as e:
            print(f"Warning: {e}")
            continue

    return images_features, images_mask, selected_keys

def load_random_gaussian_features(max_images, image_feature_dim, num_images_to_select=None, std=0.1):
    """
    Generate random gaussian features instead of using real image features.
    
    Args:
        max_images: Maximum number of images to load
        image_feature_dim: Dimension of image features
        num_images_to_select: Number of random images to generate (if None, uses max_images)
        std: Standard deviation for gaussian noise
    
    Returns:
        images_features: Tensor of random gaussian features
        images_mask: Mask indicating which positions have valid images
        feature_keys: List of generated feature identifiers
    """
    if num_images_to_select is None:
        num_images_to_select = max_images
    
    # Ensure we don't generate more images than the maximum
    num_images_to_select = min(num_images_to_select, max_images)
    
    # Preallocate tensors for features and masks
    images_features = torch.zeros(
        (max_images, image_feature_dim), dtype=torch.float32
    )
    images_mask = torch.zeros(max_images, dtype=torch.int16)
    
    # Generate random gaussian features
    feature_keys = []
    for i in range(num_images_to_select):
        # Generate random gaussian features with specified std
        random_feature = torch.randn(image_feature_dim) * std
        images_features[i] = random_feature
        images_mask[i] = 1
        feature_keys.append(f"random_gaussian_{i}")

    return images_features, images_mask, feature_keys

def load_no_image_features(max_images, image_feature_dim):
    """
    Return empty image features for text-only mode.
    
    Args:
        max_images: Maximum number of images 
        image_feature_dim: Dimension of image features
    
    Returns:
        images_features: Tensor of zeros
        images_mask: Mask with all zeros (no images)
        feature_keys: Empty list
    """
    # Return all zeros - no images
    images_features = torch.zeros(
        (max_images, image_feature_dim), dtype=torch.float32
    )
    images_mask = torch.zeros(max_images, dtype=torch.int16)
    
    return images_features, images_mask, []

def load_image_features_original(lmdb_env, image_paths, max_images, image_feature_dim):
    """
    Original function to load image features from specified paths (for comparison)
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
             do_sample=True, num_beams=3, top_p=0.9, top_k=50, repetition_penalty=1.0):
    conv = conv_templates.get(TEMPLATE_VERSION, conv_templates["v1"]).copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Check if we need to further truncate input
    model_max_length = getattr(tokenizer, 'model_max_length', 4096)
    # Setting a safety margin to leave room for generation
    max_context_length = min(model_max_length - max_new_tokens - 50, 3500)
    
    # Tokenize and check length
    raw_input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
    if len(raw_input_ids) > max_context_length:
        print(f"Warning: Input too long ({len(raw_input_ids)} tokens), truncating to {max_context_length} tokens.")
        # For simplicity, just truncate the end of the input
        input_text = tokenizer.decode(raw_input_ids[:max_context_length])
        # Regenerate prompt with truncated text
        conv = conv_templates["v1"].copy()
        # Extract just the query part without the original text
        query_prefix = query.split(": ")[0] + ": "
        truncated_query = query_prefix + "(Truncated due to length) " + input_text
        conv.append_message(conv.roles[0], truncated_query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        try:
            # KeywordsStoppingCriteria has been updated to support beam search
            outputs = model.generate(
                input_ids,
                images=images_features.unsqueeze(0).cuda(),
                do_sample=do_sample,
                temperature=temperature,
                num_beams=num_beams,  # Using the configured beam value
                top_p=top_p if do_sample else None,  # Only use top_p when sampling
                top_k=top_k if do_sample else None,  # Only use top_k when sampling
                repetition_penalty=repetition_penalty,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
        except RuntimeError as e:
            # If we get a CUDA error related to probabilities, try with more conservative parameters
            if "probability tensor" in str(e) or "CUDA" in str(e):
                print("Encountered probability tensor error. Retrying with safer parameters.")
                outputs = model.generate(
                    input_ids,
                    images=images_features.unsqueeze(0).cuda(),
                    do_sample=False,  # Disable sampling for safety
                    num_beams=1,      # Force to 1 in error recovery mode for stability
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            else:
                # Re-raise if it's a different error
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

def inference_on_dataset_ablation(model, tokenizer, config):
    """
    Enhanced inference function that supports multiple ablation types:
    - random_irrelevant_images: Use random images from LMDB
    - no_real_images: Use random gaussian features  
    - without_images: Use no image features (text-only)
    """
    data_config = config["data"]
    inference_config = config["inference"]
    ablation_config = config.get("ablation", {})
    
    # Get ablation type
    ablation_type = ablation_config.get("ablation_type", "random_irrelevant_images")
    print(f"Running ablation study with type: {ablation_type}")
    
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
    
    # Initialize LMDB environment only if needed for random_irrelevant_images
    lmdb_env = None
    all_lmdb_keys = []
    if ablation_type == "random_irrelevant_images":
        lmdb_path = data_config.get("test_lmdb_path", data_config["lmdb_path"])
        print(f"Using LMDB from: {lmdb_path}")
        lmdb_env = open_lmdb(lmdb_path)
        
        # Get all available keys from LMDB for random selection
        print("Loading all available keys from LMDB for random selection...")
        all_lmdb_keys = get_all_lmdb_keys(lmdb_env)
        print(f"Found {len(all_lmdb_keys)} images in LMDB")
    
    # Set random seed for reproducibility
    random_seed = ablation_config.get("random_seed", 42)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    print(f"Using random seed: {random_seed}")
    
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
    
    # Get beam size for filename inclusion
    num_beams = inference_config.get("num_beams", 3)
    
    print(f"Running ABLATION inference on {len(df)} samples with type: {ablation_type.upper()}")
    print("=" * 80)
    
    for idx, row in df.iterrows():
        # Extract original image paths for reference (but don't use them)
        original_image_paths = []
        if 'Image_Paths' in row:
            original_image_paths = row['Image_Paths']
        elif 'image_paths' in row:
            original_image_paths = row['image_paths']
            
        # Parse image paths if they're in string format (JSON)
        if isinstance(original_image_paths, str):
            try:
                original_image_paths = json.loads(original_image_paths)
            except:
                original_image_paths = [original_image_paths]
        
        # Determine how many images to use based on ablation type
        if ablation_type == "without_images":
            num_images_to_use = 0
        else:
            num_images_to_use = ablation_config.get("num_random_images", len(original_image_paths))
            if num_images_to_use == 0:
                num_images_to_use = data_config["max_images"]  # Use max if original had no images
        
        # Extract text content
        text_content = None
        for column in ['Body', 'body', 'text', 'content', 'Text', 'Content']:
            if column in row and pd.notna(row[column]):
                text_content = str(row[column])
                break
                
        if text_content is None:
            raise ValueError(f"No text content found for item {idx}. Required text field missing.")
            
        # Truncate text content according to max_input_len configuration
        max_input_len = data_config.get("max_input_len", 1536)
        if len(text_content) > max_input_len:
            text_content = text_content[:max_input_len]
        
        # Load features based on ablation type
        try:
            if ablation_type == "random_irrelevant_images":
                images_features, images_mask, feature_keys = load_random_image_features(
                    lmdb_env=lmdb_env,
                    all_keys=all_lmdb_keys,
                    max_images=data_config["max_images"],
                    image_feature_dim=data_config["image_feature_dim"],
                    num_images_to_select=num_images_to_use
                )
                print(f"Item {idx+1}: Using {len(feature_keys)} random irrelevant images (original had {len(original_image_paths)})")
                
            elif ablation_type == "no_real_images":
                std = ablation_config.get("random_feature_std", 0.1)
                images_features, images_mask, feature_keys = load_random_gaussian_features(
                    max_images=data_config["max_images"],
                    image_feature_dim=data_config["image_feature_dim"],
                    num_images_to_select=num_images_to_use,
                    std=std
                )
                print(f"Item {idx+1}: Using {len(feature_keys)} random gaussian features (std={std}, original had {len(original_image_paths)})")
                
            elif ablation_type == "without_images":
                images_features, images_mask, feature_keys = load_no_image_features(
                    max_images=data_config["max_images"],
                    image_feature_dim=data_config["image_feature_dim"]
                )
                print(f"Item {idx+1}: Using text-only mode (no images, original had {len(original_image_paths)})")
                
            else:
                raise ValueError(f"Unknown ablation type: {ablation_type}")
            
            has_images = len(feature_keys) > 0
            
        except Exception as e:
            print(f"Error loading features for item {idx}: {str(e)}")
            raise Exception(f"Failed to load features for item {idx}: {str(e)}")
        
        # Create prompt based on whether we have images and ablation type
        if has_images and ablation_type != "without_images":
            prompt = f"<image>\nSummarize the following article with attention to the provided images: {text_content}"
        else:
            prompt = f"Summarize the following article: {text_content}"
        
        # Generate summary
        print(f"Processing item {idx+1}/{len(df)} with ablation type '{ablation_type}'...")
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
                repetition_penalty=inference_config.get("repetition_penalty", 1.0)
            )
        except Exception as e:
            print(f"Error generating summary for item {idx}: {str(e)}")
            summary = "Error generating summary."
        
        # Store results
        result = {
            'id': row.get('id', row.get('Article_ID', idx)),
            'text': text_content[:500] + "..." if len(text_content) > 500 else text_content,
            'original_image_paths': original_image_paths,
            'ablation_type': ablation_type,
            'generated_summary': summary,
        }
        
        # Add ablation-specific fields
        if ablation_type == "random_irrelevant_images":
            result['random_image_keys'] = feature_keys
            result['num_random_images_used'] = len(feature_keys)
        elif ablation_type == "no_real_images":
            result['num_random_features_used'] = len(feature_keys)
            result['random_feature_std'] = ablation_config.get("random_feature_std", 0.1)
        elif ablation_type == "without_images":
            result['num_images_used'] = 0
            result['text_only_mode'] = True
        
        # Include reference summary if available in the dataset
        for summary_col in ['Summary', 'summary', 'reference', 'Reference']:
            if summary_col in row and pd.notna(row[summary_col]):
                result['reference_summary'] = row[summary_col]
                break
            
        results.append(result)
        
        # Create results path based on ablation type
        csv_output_path = inference_config.get("csv_output_path", "results/inference_results_ablation.csv")
        # Modify output path to indicate this is an ablation study
        base_path, ext = os.path.splitext(csv_output_path)
        csv_output_path = f"{base_path}_{ablation_type}{ext}"
        
        # Save intermediate results if configured
        if save_every_n_samples > 0 and (idx + 1) % save_every_n_samples == 0:
            save_intermediate_results(results, csv_output_path, num_beams, completed=False)
        
        print("-" * 80)
    
    # Save final results (all samples)
    save_intermediate_results(results, csv_output_path, num_beams, completed=True)
    print(f"Total samples processed: {len(results)}")
    
    # Print summary statistics based on ablation type
    if ablation_type == "random_irrelevant_images":
        print(f"Average random images per sample: {np.mean([r.get('num_random_images_used', 0) for r in results]):.2f}")
    elif ablation_type == "no_real_images":
        print(f"Average random features per sample: {np.mean([r.get('num_random_features_used', 0) for r in results]):.2f}")
    elif ablation_type == "without_images":
        print("Text-only mode: No images used for any sample")

def run_inference_ablation(config_path):
    """Main function to run ablation inference"""
    # Load configuration
    config = load_config(config_path)
    
    # Get ablation type for logging
    ablation_type = config.get("ablation", {}).get("ablation_type", "random_irrelevant_images")
    print(f"Starting ablation inference with type: {ablation_type}")
    
    # Disable gradients and set up PyTorch
    disable_torch_init()
    
    # Detect and use Flash Attention if available
    attn_implementation = get_attention_implementation(prefer_flash_attention=True)
    print(f"Using attention implementation: {attn_implementation}")
    print_attention_info(attn_implementation)
    
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
    
    # Load model with Flash Attention support
    load_kwargs = {}
    if attn_implementation and attn_implementation != "eager":
        load_kwargs["attn_implementation"] = attn_implementation
        print(f"Loading model with {attn_implementation} attention")
    
    # Note: load_pretrained_model may need to be updated to accept attn_implementation
    # For now, we'll load normally and set the implementation after
    tokenizer, model, context_len = load_pretrained_model(args, model_config["checkpoint_path"])
    
    # Set Flash Attention on the model config if supported
    if attn_implementation and hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = attn_implementation
        print(f"Set attention implementation to {attn_implementation} on loaded model")
    
    model = model.cuda()
    model.to(torch.float16)
    
    # Run ablation inference
    inference_on_dataset_ablation(model, tokenizer, config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run enhanced ablation inference with multiple ablation types")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the configuration file")
    args = parser.parse_args()
    
    # Load config to get ablation type for banner
    config = load_config(args.config)
    ablation_type = config.get("ablation", {}).get("ablation_type", "random_irrelevant_images")
    
    print("=" * 80)
    print("ENHANCED ABLATION STUDY")
    print(f"Current ablation type: {ablation_type.upper()}")
    if ablation_type == "random_irrelevant_images":
        print("This will use randomly selected images from LMDB")
        print("instead of the original relevant images for each instance.")
    elif ablation_type == "no_real_images":
        print("This will use random gaussian features")
        print("instead of real image features for each instance.")
    elif ablation_type == "without_images":
        print("This will use text-only mode")
        print("with no image features for any instance.")
    print("=" * 80)
    
    run_inference_ablation(args.config)