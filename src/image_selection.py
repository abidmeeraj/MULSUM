import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import lmdb
import pickle
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from tqdm import tqdm
import spacy
from collections import defaultdict

# Add the project root to the path
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, project_root)

from src.model.builder import load_pretrained_model
from src.config_utils import load_config
from src.utils import disable_torch_init
from src.mm_utils import tokenizer_image_token
from src.constants import IMAGE_TOKEN_INDEX
from src.mmae_evaluator import MMAEEvaluator


class ImageSelector:
    """
    Image selection module based on text-image similarity in LLaVA embedding space.
    
    Implements the baseline approach:
    1. Embed summary text as a single vector E(S) using mean-pooled token embeddings
    2. Embed each image I_i using LLaVA vision tower + projector -> E(I_i)
    3. Rank by cosine similarity between E(S) and E(I_i)
    4. Return top-K images
    """
    
    @staticmethod
    def _detect_device(requested_device: str = "auto") -> str:
        """
        Automatically detect the best available device.
        
        Args:
            requested_device: Requested device ("auto", "cuda", "cpu")
            
        Returns:
            Available device string ("cuda" or "cpu")
        """
        if requested_device == "cpu":
            return "cpu"
        elif requested_device == "cuda":
            # Try to use CUDA even if not available (original behavior)
            return "cuda"
        else:  # requested_device == "auto" or any other value
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                else:
                    return "cpu"
            except ImportError:
                # If torch is not available, fallback to CPU
                return "cpu"
    
    def __init__(self, config_path: str, device: str = "auto"):
        """
        Initialize the image selector with a fine-tuned LLaVA model.
        
        Args:
            config_path: Path to the configuration file containing model settings
            device: Device to run computations on ("auto", "cuda", or "cpu"). 
                   "auto" will automatically detect the best available device.
        """
        self.config = load_config(config_path)
        self.img_config = self.config["image_selection"]
        
        # Handle device detection - config device overrides parameter if both are specified
        config_device = self.img_config.get("device", device)
        self.device = self._detect_device(config_device)
        
        self.verbose = self.img_config.get("verbose", False)
        
        if self.verbose:
            if config_device != self.device:
                print(f"🔧 Device auto-detection: requested '{config_device}' → using '{self.device}'")
            else:
                print(f"🔧 Using device: {self.device}")
            
            # Log detailed GPU status
            self._log_gpu_status()
        
        # Get strategy configuration
        self.strategy = self.img_config.get("strategy", "S1")
        self.srl_config = self.img_config.get("srl_config", {})
        self.mmr_config = self.img_config.get("mmr_config", {})
        
        # Initialize SRL components if using S2 strategy
        if self.strategy == "S2":
            self._initialize_srl_components()
        
        # Load model components based on configuration
        self._load_model_components()
        
        # Open LMDB environments
        self.lmdb_envs = {}
        
        if self.verbose:
            print(f"ImageSelector initialized with device: {self.device}")
    
    def _initialize_srl_components(self):
        """Initialize SRL (Semantic Role Labeling) components for S2 strategy."""
        if self.verbose:
            print("🔧 Initializing SRL components for S2 strategy...")
        # TODO: Implement SRL components initialization
        # This would typically involve loading spaCy models, SRL parsers, etc.
        pass
        
    def _load_model_components(self):
        """Load the required model components for image selection."""
        if self.verbose:
            print("Loading model components for image selection...")
            
        # Initialize memory tracking
        memory_before = 0
        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / (1024**3)
            if self.verbose:
                print(f"🔍 GPU memory before model loading: {memory_before:.2f}GB")
                
        disable_torch_init()
        
        # Check if we should use a specific mm_projector for image selection
        mm_projector_path = self.img_config.get("mm_projector_path")
        
        if mm_projector_path and os.path.exists(mm_projector_path):
            if self.verbose:
                print(f"🎯 Using fine-tuned mm_projector for image selection: {mm_projector_path}")
            self._load_mm_projector_only(mm_projector_path)
        else:
            if self.verbose:
                print("🔄 Loading full model (fallback mode)")
            self._load_full_model()
            
        # Monitor GPU memory after loading
        if self.verbose and torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated() / (1024**3)
            memory_delta = memory_after - memory_before
            print(f"📈 GPU memory after model loading: {memory_after:.2f}GB (Δ{memory_delta:+.2f}GB)")
            
            if memory_delta > 0.5:  # More than 500MB
                print(f"📊 Model components loaded - using {memory_delta:.2f}GB GPU memory")
        
    def _load_mm_projector_only(self, mm_projector_path: str):
        """Load only the mm_projector weights for efficient image selection."""
        model_config = self.config["model"]
        
        # Load tokenizer for text embedding
        from transformers import AutoTokenizer
        model_base = model_config["model_name_or_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load just the mm_projector weights
        if self.verbose:
            print(f"📁 Loading mm_projector weights from: {mm_projector_path}")
        mm_projector_weights = torch.load(mm_projector_path, map_location='cpu', weights_only=True)
        
        # Create mm_projector module
        # Infer dimensions from the weights
        if 'mm_projector.weight' in mm_projector_weights:
            weight = mm_projector_weights['mm_projector.weight']
            input_dim, output_dim = weight.shape[1], weight.shape[0]
        else:
            # Default LLaVA dimensions
            input_dim, output_dim = 768, 4096
            print(f"⚠️  Could not infer dimensions from weights, using defaults: {input_dim} -> {output_dim}")
        
        # Create and load the mm_projector
        self.mm_projector = nn.Linear(input_dim, output_dim, bias='mm_projector.bias' in mm_projector_weights)
        
        # Load the weights
        state_dict = {}
        for key, value in mm_projector_weights.items():
            if key.startswith('mm_projector.'):
                clean_key = key.replace('mm_projector.', '')
                state_dict[clean_key] = value
        
        self.mm_projector.load_state_dict(state_dict)
        self.mm_projector = self.mm_projector.to(self.device)
        
        # Keep in float32 for better compatibility with LMDB features
        # Only use half precision if explicitly needed for memory
        # if self.device == "cuda":
        #     self.mm_projector = self.mm_projector.half()
        
        self.mm_projector.eval()
        
        # For text embedding, we need a simple text encoder (we'll use mean pooling of token embeddings)
        # Load a minimal language model for text embedding
        print("📝 Loading text encoder for summary embedding...")
        try:
            self._load_text_encoder(model_base)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ GPU out of memory during text encoder loading!")
                if torch.cuda.is_available():
                    memory_info = torch.cuda.mem_get_info()
                    memory_free_gb = memory_info[0] / (1024**3)
                    print(f"   GPU memory free: {memory_free_gb:.2f}GB")
                    print(f"   Consider using CPU mode: set device='cpu' in config")
                raise RuntimeError(f"GPU memory exhausted during text encoder loading: {e}")
            else:
                raise e
        
        print(f"✅ MM projector loaded: {input_dim} -> {output_dim} (dtype: {next(self.mm_projector.parameters()).dtype})")
        
    def _load_text_encoder(self, model_base: str):
        """Load a minimal text encoder for text embedding."""
        try:
            from transformers import AutoModel
            
            # Load just the embedding layers
            load_kwargs = {
                'torch_dtype': torch.float16 if self.device == "cuda" else torch.float32,
                'low_cpu_mem_usage': True
            }
            
            # Monitor GPU memory before loading
            memory_before = 0
            if torch.cuda.is_available() and self.device == "cuda":
                memory_before = torch.cuda.memory_allocated() / (1024**3)
                if self.verbose:
                    print(f"🔍 GPU memory before text encoder: {memory_before:.2f}GB")
            
            self.text_model = AutoModel.from_pretrained(model_base, **load_kwargs)
            self.text_model = self.text_model.to(self.device)
            self.text_model.eval()
            
            # Monitor GPU memory after loading
            if torch.cuda.is_available() and self.device == "cuda":
                memory_after = torch.cuda.memory_allocated() / (1024**3)
                memory_delta = memory_after - memory_before
                if self.verbose:
                    print(f"📈 GPU memory after text encoder: {memory_after:.2f}GB (Δ{memory_delta:+.2f}GB)")
                    
                # Warn if using too much memory
                if memory_delta > 2.0:  # More than 2GB
                    print(f"⚠️  Text encoder using {memory_delta:.2f}GB GPU memory")
            
            if self.verbose:
                print(f"✅ Text encoder loaded from {model_base}")
                
        except Exception as e:
            print(f"❌ Failed to load text encoder: {e}")
            if "out of memory" in str(e).lower() and torch.cuda.is_available():
                memory_info = torch.cuda.mem_get_info()
                memory_free_gb = memory_info[0] / (1024**3)
                print(f"   GPU memory free: {memory_free_gb:.2f}GB")
                print(f"   Try using CPU mode or a smaller model")
            raise
            print("✅ Text encoder loaded successfully")
        except Exception as e:
            print(f"⚠️  Could not load text encoder: {e}")
            print("Will use tokenizer embeddings as fallback")
            self.text_model = None
        
    def _load_full_model(self):
        """Load the full LLaVA model (fallback method)."""
        model_config = self.config["model"]
        
        # Create args object for model loading
        class ModelArgs:
            def __init__(self, model_base, pretrain_mm_mlp_adapter=None):
                self.model_base = model_base
                self.pretrain_mm_mlp_adapter = pretrain_mm_mlp_adapter
        
        args = ModelArgs(
            model_base=model_config["model_name_or_path"],
            pretrain_mm_mlp_adapter=model_config.get("pretrain_mm_mlp_adapter")
        )
        
        # Load the model with stage2 checkpoint (fine-tuned weights)
        self.tokenizer, self.model, self.context_len = load_pretrained_model(
            args=args,
            stage2=model_config.get("checkpoint_path")
        )
        
        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Convert to half precision for efficiency
        if self.device == "cuda":
            self.model = self.model.half()
        
        # Set mm_projector reference for compatibility
        self.mm_projector = self.model.get_model().mm_projector
        self.text_model = self.model  # Full model can do text encoding
            
        print(f"Model loaded successfully on {self.device}")
    
    def _get_lmdb_env(self, lmdb_path: str):
        """Get or create LMDB environment."""
        if lmdb_path not in self.lmdb_envs:
            self.lmdb_envs[lmdb_path] = lmdb.open(
                lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
            )
        return self.lmdb_envs[lmdb_path]
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path for LMDB key lookup."""
        return os.path.normpath(path).replace("\\", "/")
    
    def _get_image_feature_from_lmdb(self, lmdb_env, image_path: str, is_gold_image: bool = False):
        """
        Load image feature from LMDB with path normalization.
        
        Args:
            lmdb_env: LMDB environment
            image_path: Image path to load
            is_gold_image: Whether this is a gold/reference image (affects error handling)
            
        Returns:
            Image feature tensor
            
        Raises:
            KeyError: If image not found and is_gold_image=True
        """
        # Normalize path for LMDB key lookup
        normalized_path = self._normalize_path(image_path)
        
        # Try different path variations for LMDB lookup
        path_variations = [
            normalized_path,                           # Original path as-is
            image_path,                               # Original path without normalization
        ]
        
        # Only add prefix if the path doesn't already start with test_data/img/
        if not normalized_path.startswith("test_data/img/"):
            path_variations.extend([
                f"test_data/img/{normalized_path}",       # Add test/img/ prefix to normalized
                f"test_data/img/{image_path}",            # Add prefix to original (non-normalized)
            ])
        
        # Also try without the prefix if the path currently has it (for backwards compatibility)
        if normalized_path.startswith("test_data/img/"):
            base_path = normalized_path[len("test_data/img/"):]
            path_variations.append(base_path)
        if image_path.startswith("test_data/img/"):
            base_path = image_path[len("test_data/img/"):]
            path_variations.append(base_path)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for path in path_variations:
            if path not in seen:
                seen.add(path)
                unique_variations.append(path)
        
        with lmdb_env.begin() as txn:
            for attempt_path in unique_variations:
                try:
                    value = txn.get(attempt_path.encode('utf-8'))
                    if value is not None:
                        feature = pickle.loads(value)
                        feature_tensor = torch.from_numpy(feature).float()
                        if is_gold_image:
                            print(f"   ✅ Found gold image with key: '{attempt_path}'")
                        return feature_tensor
                except Exception as e:
                    if is_gold_image:
                        print(f"   ⚠️  Error loading '{attempt_path}': {e}")
                    continue
        
        # If we get here, image was not found with any path variation
        error_msg = f"Image not found in LMDB: '{image_path}'"
        if len(unique_variations) > 1:
            error_msg += f" (tried: {unique_variations})"
            
        if is_gold_image:
            # For gold images, this is a critical error
            print(f"   ❌ CRITICAL: Gold image not found in LMDB: '{image_path}'")
            print(f"      Tried variations: {unique_variations}")
            raise KeyError(f"CRITICAL: Gold image not found in LMDB: '{image_path}' (tried: {unique_variations})")
        else:
            # For candidate images, just log and raise KeyError
            if len(unique_variations) <= 3:  # Only show details for reasonable number of attempts
                print(f"   ⚠️  Candidate image not found: '{image_path}' (tried: {unique_variations})")
            raise KeyError(error_msg)
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed text using LLaVA's text encoder with mean pooling.
        
        Args:
            text: Input text to embed
            
        Returns:
            Text embedding as a tensor of shape (hidden_size,)
        """
        # Tokenize the text
        tokens = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=512
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        
        with torch.no_grad():
            if self.text_model is not None:
                # Use the text model for embedding
                try:
                    embeddings = self.text_model.get_input_embeddings()(input_ids)
                except Exception as e:
                    print(f"⚠️  Error using text model, falling back to tokenizer: {e}")
                    # Fallback to tokenizer embeddings
                    return self._embed_text_fallback(input_ids, attention_mask)
            else:
                # Use fallback method
                return self._embed_text_fallback(input_ids, attention_mask)
            
            # Mean pool across the sequence dimension (excluding special tokens if needed)
            # Apply attention mask to exclude padding tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
        return mean_embeddings.squeeze(0)  # Remove batch dimension
    
    def _embed_text_fallback(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Fallback text embedding using tokenizer embeddings.
        
        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            
        Returns:
            Text embedding tensor
        """
        # Create a simple embedding layer if we don't have the text model
        vocab_size = self.tokenizer.vocab_size
        if not hasattr(self, '_fallback_embeddings'):
            # Create a simple embedding layer (this won't be as good as the real model)
            hidden_size = 4096  # Default LLaVA hidden size
            self._fallback_embeddings = nn.Embedding(vocab_size, hidden_size)
            self._fallback_embeddings = self._fallback_embeddings.to(self.device)
            if self.device == "cuda":
                self._fallback_embeddings = self._fallback_embeddings.half()
        
        with torch.no_grad():
            embeddings = self._fallback_embeddings(input_ids)
            
            # Mean pool with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
        return mean_embeddings.squeeze(0)
    
    def embed_images(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Embed images using LLaVA's fine-tuned mm_projector.
        
        Args:
            image_features: Raw image features tensor of shape (num_images, feature_dim)
            
        Returns:
            Projected image embeddings of shape (num_images, hidden_size)
        """
        image_features = image_features.to(self.device)
        
        # Ensure dtype consistency with mm_projector
        if hasattr(self.mm_projector, 'weight'):
            target_dtype = self.mm_projector.weight.dtype
            image_features = image_features.to(dtype=target_dtype)
        
        with torch.no_grad():
            # Project through the fine-tuned mm_projector
            projected_features = self.mm_projector(image_features)
            
        return projected_features
    
    def compute_similarity(self, text_embedding: torch.Tensor, 
                          image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between text and image embeddings.
        
        Args:
            text_embedding: Text embedding of shape (hidden_size,)
            image_embeddings: Image embeddings of shape (num_images, hidden_size)
            
        Returns:
            Similarity scores of shape (num_images,)
        """
        # Normalize embeddings for cosine similarity
        text_norm = F.normalize(text_embedding.unsqueeze(0), p=2, dim=1)  # (1, hidden_size)
        image_norm = F.normalize(image_embeddings, p=2, dim=1)  # (num_images, hidden_size)
        
        # Compute cosine similarity
        similarities = torch.mm(text_norm, image_norm.t()).squeeze(0)  # (num_images,)
        
        return similarities
    
    @staticmethod
    def extract_article_id(image_path: str) -> Optional[str]:
        """
        Extract article ID from image path.
        
        Args:
            image_path: Image path like 'test_data/img/35b29c2601c2f3b5f375773303ef0cb0d525fa76_1.jpg'
                       or '35b29c2601c2f3b5f375773303ef0cb0d525fa76_1.jpg'
                       
        Returns:
            Article ID (part before the last underscore and number)
        """
        # Get just the filename without directory
        filename = os.path.basename(image_path)
        
        # Remove file extension
        name_without_ext = os.path.splitext(filename)[0]
        
        # Split by underscore and take all parts except the last (which should be image number)
        parts = name_without_ext.split('_')
        if len(parts) >= 2:
            # Join all parts except the last one (image number)
            article_id = '_'.join(parts[:-1])
            return article_id
        
        return None
    
    @staticmethod
    def filter_images_by_article(image_paths: List[str], target_article_id: str) -> List[str]:
        """
        Filter image paths to only include those from the specified article.
        
        Args:
            image_paths: List of image paths to filter
            target_article_id: Article ID to filter by
            
        Returns:
            Filtered list of image paths from the same article
        """
        filtered_paths = []
        
        for path in image_paths:
            article_id = ImageSelector.extract_article_id(path)
            if article_id == target_article_id:
                filtered_paths.append(path)
        
        return filtered_paths
    
    def select_images(self, summary_text: str, image_paths: List[str], 
                     lmdb_path: str, top_k: Optional[int] = None, 
                     similarity_threshold: Optional[float] = None,
                     selection_method: Optional[str] = None,
                     return_features: bool = False) -> Tuple[List[str], List[float]]:
        """
        Select most relevant images for the given summary text.
        
        Args:
            summary_text: Generated summary text
            image_paths: List of image paths to select from
            lmdb_path: Path to LMDB containing image features
            top_k: Number of top images to select (overrides config if provided)
            similarity_threshold: Similarity threshold (overrides config if provided)
            selection_method: Selection method (overrides config if provided)
                            Options: 'top_k', 'threshold', 'top_k_above_threshold'
            return_features: If True, return features along with paths and scores
            
        Returns:
            If return_features=False: Tuple of (selected_image_paths, similarity_scores)
            If return_features=True: Dict with 'selected_images', 'similarity_scores', 'selected_features'
        """
        # Safety check: ensure image_paths is actually a list
        if isinstance(image_paths, str):
            print(f"⚠️  WARNING: image_paths received as string instead of list: {repr(image_paths)}")
            # Try to parse it as JSON
            try:
                import json
                import ast
                # Try ast.literal_eval first (safer for Python list strings)
                if image_paths.strip().startswith("['") and image_paths.strip().endswith("']"):
                    image_paths = ast.literal_eval(image_paths.strip())
                else:
                    image_paths = json.loads(image_paths)
                print(f"✅ Successfully parsed string to list with {len(image_paths)} paths")
            except (json.JSONDecodeError, ValueError, SyntaxError) as e:
                print(f"❌ Failed to parse image_paths string: {e}")
                if return_features:
                    return {'selected_images': [], 'similarity_scores': [], 'selected_features': []}
                return [], []
        
        if not image_paths:
            if return_features:
                return {'selected_images': [], 'similarity_scores': [], 'selected_features': []}
            return [], []
        
        # Choose strategy based on configuration
        if self.strategy == "S2":
            selected_images, scores = self._select_images_s2(summary_text, image_paths, lmdb_path, top_k, 
                                        similarity_threshold, selection_method)
        elif self.strategy == "MMR":
            selected_images, scores = self._select_images_mmr(summary_text, image_paths, lmdb_path, top_k, 
                                         similarity_threshold, selection_method)
        else:
            selected_images, scores = self._select_images_s1(summary_text, image_paths, lmdb_path, top_k, 
                                        similarity_threshold, selection_method)
        
        # If features are requested, get them
        if return_features:
            selected_features = self._get_clip_features_for_images(selected_images, lmdb_path)
            return {
                'selected_images': selected_images,
                'similarity_scores': scores,
                'selected_features': selected_features
            }
        
        return selected_images, scores
    
    def select_images_with_features(self, summary_text: str, image_paths: List[str], 
                                   lmdb_path: str, num_select: int = 3) -> Dict[str, Any]:
        """
        Convenience method that selects images and returns both paths and features.
        Designed for integration with MMAE evaluator.
        
        Args:
            summary_text: Generated summary text
            image_paths: List of available image paths
            lmdb_path: Path to LMDB containing CLIP features
            num_select: Number of images to select
            
        Returns:
            Dictionary containing:
                - 'selected_images': List of selected image paths
                - 'similarity_scores': List of similarity scores
                - 'selected_features': List of CLIP feature tensors
        """
        return self.select_images(
            summary_text=summary_text,
            image_paths=image_paths,
            lmdb_path=lmdb_path,
            top_k=num_select,
            return_features=True
        )
        
    def _select_images_s1(self, summary_text: str, image_paths: List[str], 
                         lmdb_path: str, top_k: Optional[int] = None,
                         similarity_threshold: Optional[float] = None,
                         selection_method: Optional[str] = None) -> Tuple[List[str], List[float]]:
        """
        S1 Strategy: Whole summary embedding approach.
        Embed summary text as a single vector and rank images by cosine similarity.
        """
        if self.verbose:
            print(f"🔍 Using S1 strategy: whole summary embedding")
        
        # Always filter images by article to ensure consistency
        original_count = len(image_paths)
        article_ids = set()
        for path in image_paths:
            article_id = self.extract_article_id(path)
            if article_id:
                article_ids.add(article_id)
        
        if len(article_ids) > 1:
            # Multiple articles detected - filter to the most common one
            from collections import Counter
            article_counts = Counter()
            for path in image_paths:
                article_id = self.extract_article_id(path)
                if article_id:
                    article_counts[article_id] += 1
            
            if article_counts:
                target_article = article_counts.most_common(1)[0][0]
                image_paths = self.filter_images_by_article(image_paths, target_article)
                if self.verbose:
                    print(f"🔍 Multiple articles detected, filtered to '{target_article}': {len(image_paths)}/{original_count} images")
                
                if not image_paths:
                    if self.verbose:
                        print(f"⚠️  No images found for article '{target_article}' after filtering")
                    return [], []
        elif len(article_ids) == 1:
            if self.verbose:
                print(f"🔍 Single article detected: {list(article_ids)[0]}")
        
        # Get LMDB environment
        lmdb_env = self._get_lmdb_env(lmdb_path)
        
        # Embed the summary text once
        text_embedding = self.embed_text(summary_text)
        
        # Load and embed all images
        valid_paths = []
        image_embeddings = []
        
        
        missing_count = 0
        if self.verbose:
            print(f"Loading {len(image_paths)} images from LMDB...")
        for path in tqdm(image_paths, desc="Loading images", disable=not self.verbose):
            try:
                image_feature = self._get_image_feature_from_lmdb(lmdb_env, path)
                # Project image features through mm_projector to get embeddings
                with torch.no_grad():
                    image_embedding = self.mm_projector(image_feature.unsqueeze(0).to(self.device))
                    image_embeddings.append(image_embedding.squeeze(0))
                valid_paths.append(path)
            except KeyError as e:
                missing_count += 1
                if self.verbose and missing_count <= 3:  # Only show first 3 missing images
                    print(f"⚠️  Candidate image not found: '{path}'")
                continue
        
        if missing_count > 0 and not self.verbose:
            print(f"⚠️  {missing_count} candidate images not found in LMDB")
        elif missing_count > 3 and self.verbose:
            print(f"⚠️  ... and {missing_count - 3} more images not found")
            
        if not image_embeddings:
            print("No valid images found!")
            return [], []
        
        # Stack image embeddings
        image_embeddings = torch.stack(image_embeddings)
        
        # Compute similarities between text and images
        similarities = self.compute_similarity(text_embedding, image_embeddings)
        
        # Get parameters from config or arguments
        top_k = top_k or self.img_config.get("top_k", 5)
        threshold = similarity_threshold or self.img_config.get("similarity_threshold", 0.0)
        method = selection_method or self.img_config.get("selection_method", "top_k")
        
        # Convert to numpy for easier processing
        similarities_np = similarities.cpu().numpy()
        
        # Apply selection method
        if method == "top_k":
            # Select top-k images by similarity
            top_indices = np.argsort(similarities_np)[::-1][:top_k]
            selected_paths = [valid_paths[i] for i in top_indices]
            selected_scores = [similarities_np[i] for i in top_indices]
            
        elif method == "threshold":
            # Select all images above threshold
            above_threshold = similarities_np >= threshold
            selected_indices = np.where(above_threshold)[0]
            # Sort by similarity (descending)
            sorted_indices = selected_indices[np.argsort(similarities_np[selected_indices])[::-1]]
            selected_paths = [valid_paths[i] for i in sorted_indices]
            selected_scores = [similarities_np[i] for i in sorted_indices]
            
        elif method == "top_k_above_threshold":
            # Select top-k images that are above threshold
            above_threshold = similarities_np >= threshold
            if not np.any(above_threshold):
                selected_paths, selected_scores = [], []
            else:
                valid_indices = np.where(above_threshold)[0]
                valid_similarities = similarities_np[valid_indices]
                # Sort and take top-k
                top_valid_indices = np.argsort(valid_similarities)[::-1][:top_k]
                final_indices = valid_indices[top_valid_indices]
                selected_paths = [valid_paths[i] for i in final_indices]
                selected_scores = [similarities_np[i] for i in final_indices]
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        if self.verbose:
            print(f"🎯 S1 selection completed: {len(selected_paths)} images selected")
            if selected_scores:
                print(f"   Similarity range: {min(selected_scores):.3f} - {max(selected_scores):.3f}")
        
        return selected_paths, selected_scores

    def _select_images_s2(self, summary_text: str, image_paths: List[str], 
                         lmdb_path: str, top_k: Optional[int] = None,
                         similarity_threshold: Optional[float] = None,
                         selection_method: Optional[str] = None) -> Tuple[List[str], List[float]]:
        """
        S2 Strategy: Semantic Role Labeling approach.
        Parse summary into semantic roles and match with image propositions.
        """
        if self.verbose:
            print(f"🔍 Using S2 strategy: Semantic Role Labeling")
        
        # TODO: Implement S2 strategy
        # For now, fall back to S1 strategy
        print("⚠️  S2 strategy not fully implemented yet, falling back to S1")
        return self._select_images_s1(summary_text, image_paths, lmdb_path, top_k, 
                                    similarity_threshold, selection_method)

    def _select_images_mmr(self, summary_text: str, image_paths: List[str], 
                          lmdb_path: str, top_k: Optional[int] = None,
                          similarity_threshold: Optional[float] = None,
                          selection_method: Optional[str] = None) -> Tuple[List[str], List[float]]:
        """
        MMR Strategy: Maximal Marginal Relevance approach.
        Balances relevance and diversity using the formula:
        d* = argmax[λ × sim(d_i, S) - (1-λ) × max(sim(d_i, d_j) for d_j in R)]
        
        Note: MMR has its own built-in selection algorithm. The similarity_threshold 
        and selection_method parameters are ignored since MMR always selects exactly 
        top_k images using its iterative relevance-diversity optimization.
        """
        if self.verbose:
            print(f"🔍 Using MMR strategy: Maximal Marginal Relevance")
            if selection_method and selection_method != "top_k":
                print(f"   ℹ️  Note: selection_method '{selection_method}' ignored in MMR (always uses top_k)")
            if similarity_threshold and similarity_threshold > 0:
                print(f"   ℹ️  Note: similarity_threshold ignored in MMR (uses built-in selection)")
        
        # Get MMR parameters from config
        lambda_param = self.mmr_config.get("lambda", 0.7)
        similarity_function = self.mmr_config.get("similarity_function", "cosine")
        
        if self.verbose:
            print(f"   λ (lambda): {lambda_param}")
            print(f"   Similarity function: {similarity_function}")
        
        # Always filter images by article to ensure consistency
        original_count = len(image_paths)
        article_ids = set()
        for path in image_paths:
            article_id = self.extract_article_id(path)
            if article_id:
                article_ids.add(article_id)
        
        if len(article_ids) > 1:
            # Multiple articles detected - filter to the most common one
            from collections import Counter
            article_counts = Counter()
            for path in image_paths:
                article_id = self.extract_article_id(path)
                if article_id:
                    article_counts[article_id] += 1
            
            if article_counts:
                target_article = article_counts.most_common(1)[0][0]
                image_paths = self.filter_images_by_article(image_paths, target_article)
                if self.verbose:
                    print(f"🔍 Multiple articles detected, filtered to '{target_article}': {len(image_paths)}/{original_count} images")
                
                if not image_paths:
                    if self.verbose:
                        print(f"⚠️  No images found for article '{target_article}' after filtering")
                    return [], []
        elif len(article_ids) == 1:
            if self.verbose:
                print(f"🔍 Single article detected: {list(article_ids)[0]}")
        
        # Get LMDB environment
        lmdb_env = self._get_lmdb_env(lmdb_path)
        
        # Embed the summary text once
        text_embedding = self.embed_text(summary_text)
        
        # Load and embed all images
        valid_paths = []
        image_embeddings = []
        
        missing_count = 0
        if self.verbose:
            print(f"Loading {len(image_paths)} images from LMDB...")
        for path in tqdm(image_paths, desc="Loading images", disable=not self.verbose):
            try:
                image_feature = self._get_image_feature_from_lmdb(lmdb_env, path)
                # Project image features through mm_projector to get embeddings
                with torch.no_grad():
                    image_embedding = self.mm_projector(image_feature.unsqueeze(0).to(self.device))
                    image_embeddings.append(image_embedding.squeeze(0))
                valid_paths.append(path)
            except KeyError as e:
                missing_count += 1
                if self.verbose and missing_count <= 3:  # Only show first 3 missing images
                    print(f"⚠️  Candidate image not found: '{path}'")
                continue
        
        if missing_count > 0 and not self.verbose:
            print(f"⚠️  {missing_count} candidate images not found in LMDB")
        elif missing_count > 3 and self.verbose:
            print(f"⚠️  ... and {missing_count - 3} more images not found")
            
        if not image_embeddings:
            print("No valid images found!")
            return [], []
        
        # Stack image embeddings
        image_embeddings = torch.stack(image_embeddings)
        
        # Pre-compute all similarities between text and images (relevance scores)
        if similarity_function == "cosine":
            relevance_scores = F.cosine_similarity(
                text_embedding.unsqueeze(0), 
                image_embeddings, 
                dim=1
            ).cpu().numpy()
        elif similarity_function == "dot_product":
            relevance_scores = torch.mm(
                text_embedding.unsqueeze(0), 
                image_embeddings.t()
            ).squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"Unknown similarity function: {similarity_function}")
        
        # Use provided parameters or fall back to config
        top_k = top_k or self.img_config.get("top_k", 5)
        
        # MMR Selection Algorithm
        selected_indices = []
        selected_paths = []
        selected_scores = []
        remaining_indices = list(range(len(valid_paths)))
        
        if self.verbose:
            print(f"🔄 Starting MMR selection for top-{top_k} images...")
        
        for step in range(min(top_k, len(valid_paths))):
            if not remaining_indices:
                break
                
            best_score = float('-inf')
            best_idx = None
            best_idx_in_remaining = None
            
            # For each remaining candidate
            for i, candidate_idx in enumerate(remaining_indices):
                # Relevance term: similarity with summary
                relevance = relevance_scores[candidate_idx]
                
                # Diversity term: maximum similarity with already selected images
                if selected_indices:
                    # Compute similarities with all selected images
                    candidate_embedding = image_embeddings[candidate_idx].unsqueeze(0)
                    selected_embeddings = image_embeddings[selected_indices]
                    
                    if similarity_function == "cosine":
                        diversities = F.cosine_similarity(
                            candidate_embedding, 
                            selected_embeddings, 
                            dim=1
                        ).cpu().numpy()
                    elif similarity_function == "dot_product":
                        diversities = torch.mm(
                            candidate_embedding, 
                            selected_embeddings.t()
                        ).squeeze(0).cpu().numpy()
                    
                    max_diversity = np.max(diversities)
                else:
                    max_diversity = 0.0  # No penalty for first selection
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = candidate_idx
                    best_idx_in_remaining = i
            
            # Add best candidate to selected set
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_paths.append(valid_paths[best_idx])
                selected_scores.append(relevance_scores[best_idx])  # Store original relevance score
                remaining_indices.pop(best_idx_in_remaining)
                
                if self.verbose and step < 3:  # Show details for first few selections
                    print(f"   Step {step+1}: {os.path.basename(valid_paths[best_idx])} "
                          f"(relevance: {relevance_scores[best_idx]:.3f}, MMR score: {best_score:.3f})")
        
        if self.verbose:
            print(f"🎯 MMR selection completed: {len(selected_paths)} images selected")
            if selected_scores:
                print(f"   Relevance range: {min(selected_scores):.3f} - {max(selected_scores):.3f}")
        
        return selected_paths, selected_scores

    def evaluate_selection_with_mmae(self, 
                                   summary_text: str,
                                   reference_summary: str,
                                   candidate_images: List[str], 
                                   gold_images: List[str],
                                   lmdb_path: str,
                                   top_k: Optional[int] = None,
                                   similarity_threshold: Optional[float] = None,
                                   selection_method: Optional[str] = None,
                                   clip_model_name: str = "openai/clip-vit-base-patch16") -> Dict[str, Any]:
        """
        Evaluate image selection with both traditional metrics and MMAE.
        
        Args:
            summary_text: Generated summary text
            reference_summary: Reference/gold summary text  
            candidate_images: List of all candidate image paths
            gold_images: List of ground truth relevant image paths
            lmdb_path: Path to LMDB containing image features
            top_k: Number of top images to select
            similarity_threshold: Similarity threshold for selection
            selection_method: Selection method to use
            clip_model_name: CLIP model for MMAE evaluation
            
        Returns:
            Dictionary containing both traditional and MMAE evaluation results
        """
        # Perform image selection
        selected_images, selection_scores = self.select_images(
            summary_text=summary_text,
            image_paths=candidate_images,
            lmdb_path=lmdb_path,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            selection_method=selection_method
        )
        
        # Calculate traditional metrics
        traditional_results = self._calculate_traditional_metrics(selected_images, gold_images)
        
        # Calculate MMAE if we have selected images
        mmae_results = {}
        if selected_images:
            try:
                # Get MMAE configuration from config
                mmae_config = self.config.get("mmae_config", {})
                
                # Initialize MMAE evaluator with configuration
                mmae_evaluator = MMAEEvaluator(
                    clip_model_name=mmae_config.get("clip_model_name", clip_model_name),
                    device=self.device,
                    use_sentence_level_maxsim=mmae_config.get("use_sentence_level_maxsim", True),
                    verbose_sentence_processing=mmae_config.get("verbose_sentence_processing", False),
                    rouge_metric=mmae_config.get("rouge_metric", "rouge_l")
                )
                
                # Get CLIP features for selected images
                selected_image_features = self._get_clip_features_for_images(selected_images, lmdb_path)
                
                # Calculate MMAE
                mmae_results = mmae_evaluator.evaluate_article(
                    reference_summary=reference_summary,
                    generated_summary=summary_text,
                    selected_images=selected_images,
                    selected_image_features=selected_image_features,
                    gold_images=gold_images
                )
                
                print(f"📊 MMAE Evaluation Results:")
                print(f"   ROUGE-L: {mmae_results['rouge_l']:.3f}")
                print(f"   Max Image-Text Similarity: {mmae_results['max_similarity']:.3f}")
                print(f"   Image Precision: {mmae_results['image_precision']:.3f}")
                print(f"   MMAE Score: {mmae_results['mmae']:.3f}")
                
            except Exception as e:
                print(f"⚠️  Error calculating MMAE: {e}")
                mmae_results = {'error': str(e)}
        else:
            mmae_results = {'error': 'No images selected'}
        
        # Combine results
        results = {
            'strategy': self.strategy,
            'selected_images': selected_images,
            'selection_scores': selection_scores,
            'traditional_metrics': traditional_results,
            'mmae_metrics': mmae_results,
            'config': {
                'top_k': top_k or self.img_config.get('top_k', 5),
                'similarity_threshold': similarity_threshold or self.img_config.get('similarity_threshold', 0.0),
                'selection_method': selection_method or self.img_config.get('selection_method', 'top_k'),
                'clip_model': clip_model_name
            }
        }
        
        return results
    
    def _calculate_traditional_metrics(self, selected_images: List[str], gold_images: List[str]) -> Dict[str, float]:
        """Calculate traditional precision, recall, F1 metrics."""
        if not selected_images and not gold_images:
            return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
        
        if not selected_images:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        if not gold_images:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        selected_set = set(selected_images)
        gold_set = set(gold_images)
        
        intersection = selected_set.intersection(gold_set)
        
        precision = len(intersection) / len(selected_set) if selected_set else 0.0
        recall = len(intersection) / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall, 
            'f1': f1,
            'intersection_count': len(intersection),
            'selected_count': len(selected_set),
            'gold_count': len(gold_set)
        }
    
    def _get_clip_features_for_images(self, image_paths: List[str], lmdb_path: str) -> List[torch.Tensor]:
        """
        Get raw CLIP vision features for selected images from LMDB.
        
        Args:
            image_paths: List of image paths
            lmdb_path: Path to LMDB containing raw CLIP vision features
            
        Returns:
            List of raw CLIP vision feature tensors (768-dim for patch16, before projection)
        """
        lmdb_env = self._get_lmdb_env(lmdb_path)
        clip_features = []
        
        for image_path in image_paths:
            try:
                # Get the raw CLIP vision feature (should be 768-dim for patch16, before vision projection)
                clip_feature = self._get_image_feature_from_lmdb(lmdb_env, image_path)
                
                # Ensure it's the right dimension (768 for patch16 vision features before projection)
                if clip_feature.shape[0] != 768:
                    print(f"⚠️  Warning: Expected 768-dim raw CLIP vision features, got {clip_feature.shape[0]}-dim for {image_path}")
                    print(f"    MMAE expects raw vision features before the vision projection layer.")
                    
                clip_features.append(clip_feature)
                
            except KeyError:
                print(f"⚠️  Could not find CLIP features for image: {image_path}")
                # Skip missing images rather than failing
                continue
        
        return clip_features

    def close(self):
        """Close LMDB environments."""
        for env in self.lmdb_envs.values():
            env.close()
        self.lmdb_envs.clear()


def load_all_lmdb_keys(lmdb_path: str) -> List[str]:
    """
    Load all keys from an LMDB database.
    
    Args:
        lmdb_path: Path to the LMDB database
        
    Returns:
        List of all keys in the database
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
    keys = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            keys.append(key.decode())
    
    env.close()
    return keys


def run_image_selection_evaluation(config_path: str, summary_text: str, 
                                 candidate_images: List[str], gold_images: List[str],
                                 output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run complete image selection evaluation.
    
    Args:
        config_path: Path to configuration file
        summary_text: Generated summary text
        candidate_images: List of all candidate image paths
        gold_images: List of ground truth relevant image paths
        output_path: Optional path to save results
        
    Returns:
        Evaluation results dictionary
    """
    # Initialize image selector
    selector = ImageSelector(config_path)
    
    try:
        # Get LMDB path from config
        config = load_config(config_path)
        lmdb_path = config["data"]["test_lmdb_path"]
        
        # Run evaluation
        results = selector.evaluate_selection(
            summary_text=summary_text,
            candidate_images=candidate_images,
            gold_images=gold_images,
            lmdb_path=lmdb_path
        )
        
        # Add configuration info to results
        results["config_path"] = config_path
        results["summary_text"] = summary_text
        results["timestamp"] = pd.Timestamp.now().isoformat()
        
        # Save results if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
            
        return results
        
    finally:
        selector.close()