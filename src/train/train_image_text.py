import os
import sys
import torch
import argparse
import pandas as pd
from dataclasses import dataclass, field
import pathlib
from typing import Dict, Optional, Sequence
import logging
import csv
import wandb

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.train.train import ModelArguments, TrainingArguments, maybe_zero_3, get_peft_state_maybe_zero_3
from src.train.train import get_peft_state_non_lora_maybe_zero_3, get_mm_adapter_state_maybe_zero_3
from src.train.train import find_all_linear_names, safe_save_model_for_hf_trainer, smart_tokenizer_and_embedding_resize
from src.train.train import rank0_print
from src.train.dataset import make_supervised_data_module, DataArguments
from src import conversation as conversation_lib
from src.model.builder import load_pretrained_model
from src.model import MulsumLlamaForCausalLM  # Import our custom model class
from src.mm_utils import print_trainable_parameters
from src.config_utils import load_config, config_to_args

from transformers import Trainer, TrainingArguments as HfTrainingArguments
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

# Enable Flash Attention backend early
if torch.cuda.is_available():
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("Flash Attention backend enabled")
    except Exception as e:
        print(f"Failed to enable Flash Attention backend: {e}")

local_rank = None

# Default configuration path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                   "configs", "image_text_config.json")

def get_local_model_path(model_base):
    """Get local model path if it exists, otherwise return the original path"""
    # Check if it's already a local path
    if os.path.exists(model_base):
        return model_base
    
    # Check for local models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    local_model_path = os.path.join(models_dir, model_base.replace("/", "_"))
    
    if os.path.exists(local_model_path):
        print(f"Using local model from: {local_model_path}")
        return local_model_path
    else:
        print(f"Local model not found at {local_model_path}, using online: {model_base}")
        return model_base

def setup_wandb(config, training_args, model_args, data_args):
    """Initialize Weights & Biases tracking"""
    wandb_config = config.get("wandb", {})
    
    if not wandb_config.get("enabled", False):
        rank0_print("Weights & Biases tracking is disabled")
        return None
    
    # Only initialize wandb on rank 0 to avoid multiple runs
    if local_rank != 0 and local_rank != -1:
        return None
    
    try:
        # Generate run name if not provided
        run_name = wandb_config.get("name")
        if run_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"mulsum-{timestamp}"
        
        # Initialize wandb
        wandb.init(
            project=wandb_config.get("project", "mulsum-training"),
            name=run_name,
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
            config={
                # Model configuration
                "model_name": model_args.model_name_or_path,
                "version": model_args.version,
                "trainable_mm_projector": model_args.trainable_mm_projector,
                
                # Training configuration
                "num_train_epochs": training_args.num_train_epochs,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "fp16": training_args.fp16,
                "bf16": training_args.bf16,
                "weight_decay": training_args.weight_decay,
                
                # Data configuration
                "max_images": data_args.max_images,
                "image_feature_dim": data_args.image_feature_dim,
                "max_input_len": data_args.max_input_len,
                "max_summary_len": data_args.max_summary_len,
                "train_subset_size": data_args.train_subset_size,
                "validation_subset_size": data_args.validation_subset_size,
                
                # LoRA configuration if enabled
                **{k: v for k, v in config.get("training", {}).items() if k.startswith("lora_")},
                
                # Additional training config
                "bits": config.get("training", {}).get("bits", 16),
                "eval_strategy": training_args.eval_strategy,
                "eval_steps": training_args.eval_steps,
                "save_steps": training_args.save_steps,
            }
        )
        
        rank0_print(f"✓ Weights & Biases initialized - Project: {wandb_config.get('project')} | Run: {run_name}")
        rank0_print(f"  View training at: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
        
        return wandb_config
        
    except Exception as e:
        rank0_print(f"Warning: Failed to initialize Weights & Biases: {e}")
        rank0_print("Training will continue without wandb tracking")
        return None

def train_image_text_summarization(config_path=DEFAULT_CONFIG_PATH):
    global local_rank
    
    # Load configuration from file
    config = load_config(config_path)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Convert config sections to appropriate argument objects
    model_args = ModelArguments(
        model_name_or_path=config["model"]["model_name_or_path"],
        pretrain_mm_mlp_adapter=config["model"]["pretrain_mm_mlp_adapter"],
        version=config["model"].get("version", "v1"),
        trainable_mm_projector=config["model"].get("trainable_mm_projector", False)
    )
    
    # Create HuggingFace training arguments with evaluation settings
    training_config = config["training"]
    data_config = config["data"]
    
    # Get model max length from data config (use max_input_len) or use default
    model_max_length = data_config.get("max_input_len", 2048)
    rank0_print(f"Using model_max_length: {model_max_length} (from config max_input_len)")
    
    # Add validation settings to data args, including train and validation subset sizes
    data_args = DataArguments(
        data_path=config["data"]["data_path"],
        validation_path=config["data"].get("validation_path", None),
        lmdb_path=config["data"]["lmdb_path"],
        validation_lmdb_path=config["data"].get("validation_lmdb_path", None),
        max_images=config["data"]["max_images"],
        image_feature_dim=config["data"]["image_feature_dim"],
        max_input_len=model_max_length,  # Use the same value for consistency
        max_summary_len=config["data"]["max_summary_len"],
        train_subset_size=config["data"].get("train_subset_size", 0),
        validation_subset_size=config["data"].get("validation_subset_size", 0),
        dataloader_num_workers=training_config.get("dataloader_num_workers", 0),
    )
    training_args = HfTrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 
                                                      training_config["per_device_train_batch_size"]),
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        logging_steps=10,
        # Evaluation and checkpoint saving settings
        eval_strategy=training_config.get("eval_strategy", "no"),
        eval_steps=training_config.get("eval_steps", None),
        save_strategy=training_config.get("save_strategy", "epoch"),
        save_steps=training_config.get("save_steps", None),
        save_total_limit=training_config.get("save_total_limit", 3),
        # Load best model at the end of training
        metric_for_best_model=training_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=training_config.get("greater_is_better", False),
        load_best_model_at_end=training_config.get("load_best_model_at_end", False),
        # Other settings
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="adamw_torch",
        weight_decay=training_config.get("weight_decay", 0.0),
        # Memory optimization settings
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        deepspeed=training_config.get("deepspeed", None),
        # Prevent memory leaks by emptying cache at the end of each iteration
        max_grad_norm=1.0,
        # DDP settings for LoRA compatibility
        ddp_find_unused_parameters=False,  # Critical for LoRA to avoid the "marked ready twice" error
        # Data loading settings
        dataloader_num_workers=training_config["dataloader_num_workers"],
        dataloader_prefetch_factor=training_config["dataloader_prefetch_factor"],
        dataloader_pin_memory=False,  # Set to False for better DDP compatibility
        dataloader_persistent_workers=training_config["dataloader_persistent_workers"],
    )
    
    compute_dtype = (torch.float16 if training_args.fp16 else 
                     (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    # Model loading and configuration
    bnb_model_from_pretrained_args = {}
    bits = training_config.get("bits", 16)
    if bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        ))
    
    # Detect the best available attention implementation
    from src.utils import get_attention_implementation, print_attention_info
    attn_implementation = get_attention_implementation(prefer_flash_attention=True)
    rank0_print(f"Detected attention implementation: {attn_implementation}")
    print_attention_info(attn_implementation)
    
    # Load base model using our custom MulsumLlamaForCausalLM class
    # Get local model path if available
    local_model_path = get_local_model_path(model_args.model_name_or_path)
    
    model_load_kwargs = {
        "pretrained_model_name_or_path": local_model_path,
        "cache_dir": None,
        "low_cpu_mem_usage": True,
        "torch_dtype": compute_dtype,
    }
    
    # Add Flash Attention implementation if supported
    if attn_implementation and attn_implementation != "eager":
        model_load_kwargs["attn_implementation"] = attn_implementation
        rank0_print(f"Loading model with {attn_implementation} attention")
    
    # Add local_files_only if using local model
    if local_model_path and os.path.exists(local_model_path) and local_model_path != model_args.model_name_or_path:
        model_load_kwargs["local_files_only"] = True
    
    # Add quantization args if needed
    model_load_kwargs.update(bnb_model_from_pretrained_args)
    
    rank0_print(f"Loading base model from {local_model_path}")
    model = MulsumLlamaForCausalLM.from_pretrained(**model_load_kwargs)
    
    model.config.use_cache = False
    
    # Make sure Flash Attention was properly enabled in the model config
    if attn_implementation and hasattr(model.config, "attn_implementation"):
        if model.config.attn_implementation == attn_implementation:
            rank0_print(f"Successfully enabled {attn_implementation} for faster training and reduced memory usage")
        else:
            rank0_print(f"Warning: Attempted to use {attn_implementation} but model is using {model.config.attn_implementation}")
    
    # Initialize vision modules
    model.get_model().initialize_vision_modules(model_args=model_args)
    
    # Prepare model for k-bit training if needed
    if bits in [4, 8]:
        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    
    # Enable gradient checkpointing for memory efficiency
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Configure LoRA if enabled
    lora_enable = training_config.get("lora_enable", False)
    
    # Convert model precision before LoRA setup
    if bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    
    if lora_enable:
        lora_config = LoraConfig(
            r=training_config.get("lora_r", 64),
            lora_alpha=training_config.get("lora_alpha", 16),
            target_modules=find_all_linear_names(model),
            lora_dropout=training_config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)
        
        # Prepare LoRA model for better DDP compatibility
        rank0_print("Preparing LoRA model for DDP compatibility...")
        
        # Ensure all LoRA parameters are properly initialized
        for name, param in model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                if param.grad is not None:
                    param.grad.zero_()
        
        # Set use_cache to False for training (important for DDP)
        if hasattr(model, 'config'):
            model.config.use_cache = False
    
    # Load tokenizer and configure conversation template
    # Use the same local model path for tokenizer
    tokenizer_kwargs = {
        "cache_dir": None,
        "model_max_length": model_max_length,
        "padding_side": "right",
        "use_fast": False,
    }
    
    # Add local_files_only if using local model
    if local_model_path and os.path.exists(local_model_path) and local_model_path != model_args.model_name_or_path:
        tokenizer_kwargs["local_files_only"] = True
    
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        **tokenizer_kwargs
    )
    
    # Set pad token if not already configured (modern approach)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set the default conversation template
    rank0_print(f"Setting conversation template with version '{model_args.version}'")
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        rank0_print(f"Using conversation template: {model_args.version} (found in templates)")
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    # Move the mm_projector to the appropriate precision
    if bits in [4, 8] and hasattr(model, "get_model") and hasattr(model.get_model(), "mm_projector"):
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
    
    # Prepare datasets and trainer
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Initialize Weights & Biases tracking
    wandb_config = setup_wandb(config, training_args, model_args, data_args)
    
    # Update training args to include wandb if enabled
    if wandb_config and wandb_config.get("enabled", False):
        training_args.report_to = ["wandb", "tensorboard"]
    else:
        training_args.report_to = ["tensorboard"]
    
    # Create a custom callback to clear CUDA cache periodically
    from transformers.trainer_callback import TrainerCallback, EarlyStoppingCallback
    
    class CUDACacheCleanupCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:  # Clear cache every 100 steps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    class LossTrackingCallback(TrainerCallback):
        """Custom callback to track and save training and validation losses."""
        
        def __init__(self, output_dir):
            self.output_dir = output_dir
            self.loss_file = os.path.join(output_dir, "loss_history.csv")
            self.losses = []
            
            # If we're resuming training, load existing loss history
            if os.path.exists(self.loss_file):
                try:
                    self.losses = pd.read_csv(self.loss_file).to_dict('records')
                    rank0_print(f"Loaded existing loss history from {self.loss_file} with {len(self.losses)} records")
                except Exception as e:
                    rank0_print(f"Warning: Could not load existing loss file: {e}")
                    self.losses = []
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            """Record validation metrics whenever evaluation happens"""
            if metrics:
                # Extract current epoch, may be fractional
                current_epoch = state.epoch
                
                # Get evaluation loss
                eval_loss = metrics.get("eval_loss", None)
                
                if eval_loss is not None:
                    # Check if we already have a record for this epoch
                    existing_entry = next((item for item in self.losses if item.get("epoch") == current_epoch), None)
                    
                    if existing_entry:
                        # Update existing entry
                        existing_entry["eval_loss"] = eval_loss
                    else:
                        # Create new entry with eval_loss
                        self.losses.append({
                            "epoch": current_epoch,
                            "step": state.global_step,
                            "eval_loss": eval_loss,
                            "train_loss": None  # Will be filled when training loss is reported
                        })
                    
                    # Save updated losses
                    self._save_losses()
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Track training loss from logs"""
            if logs and "loss" in logs and "epoch" in logs:
                current_epoch = logs["epoch"]
                train_loss = logs["loss"]
                
                # Check if we already have a record for this epoch 
                existing_entry = next((item for item in self.losses if item.get("epoch") == current_epoch), None)
                
                if existing_entry:
                    # Update existing entry with train loss
                    existing_entry["train_loss"] = train_loss
                else:
                    # Create new entry with just train_loss
                    self.losses.append({
                        "epoch": current_epoch, 
                        "step": state.global_step,
                        "train_loss": train_loss,
                        "eval_loss": None  # Will be filled when evaluation happens
                    })
                
                # Save updated losses
                self._save_losses()
        
        def _save_losses(self):
            """Save losses to CSV file"""
            if not self.losses:
                return
                
            # Convert to DataFrame and save
            df = pd.DataFrame(self.losses)
            
            # Ensure the output directory exists
            os.makedirs(os.path.dirname(self.loss_file), exist_ok=True)
            
            # Save to CSV
            df.to_csv(self.loss_file, index=False)
            rank0_print(f"Loss history updated at {self.loss_file}")
    
    class WandbCallback(TrainerCallback):
        """Enhanced Weights & Biases callback for detailed logging"""
        
        def __init__(self, wandb_config=None):
            self.wandb_config = wandb_config or {}
            self.log_model = self.wandb_config.get("log_model", "end")
            self.watch_model = self.wandb_config.get("watch_model", True)
            self.log_gradients = self.wandb_config.get("log_gradients", False)
            self.log_parameters = self.wandb_config.get("log_parameters", False)
            self.model_watched = False
            
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            """Set up model watching at the beginning of training"""
            if wandb.run is None or self.model_watched:
                return
                
            if self.watch_model:
                try:
                    # Watch model for gradients and parameters
                    log_freq = 100  # Log every 100 steps to avoid overwhelming wandb
                    wandb.watch(
                        model, 
                        log="all" if self.log_gradients else None,
                        log_freq=log_freq,
                        log_graph=True
                    )
                    self.model_watched = True
                    rank0_print("✓ Model watching enabled in Weights & Biases")
                except Exception as e:
                    rank0_print(f"Warning: Could not set up model watching: {e}")
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            """Enhanced logging to wandb"""
            if wandb.run is None or not logs:
                return
                
            # Only log from rank 0
            if local_rank != 0 and local_rank != -1:
                return
                
            # Calculate training speed metrics
            if hasattr(state, 'log_history') and len(state.log_history) > 1:
                current_step = state.global_step
                current_time = state.log_history[-1].get('train_runtime', 0)
                
                # Calculate steps per second
                if current_step > 0 and current_time > 0:
                    steps_per_second = current_step / current_time
                    logs['steps_per_second'] = steps_per_second
                    
                    # Estimate time remaining
                    total_steps = state.max_steps if state.max_steps > 0 else (args.num_train_epochs * state.num_examples // args.train_batch_size)
                    remaining_steps = total_steps - current_step
                    if steps_per_second > 0:
                        estimated_remaining_time = remaining_steps / steps_per_second
                        logs['estimated_remaining_hours'] = estimated_remaining_time / 3600
            
            # Add GPU memory usage if available
            if torch.cuda.is_available():
                try:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB
                    logs['gpu_memory_allocated_gb'] = memory_allocated
                    logs['gpu_memory_cached_gb'] = memory_cached
                except:
                    pass
            
            # Log to wandb
            try:
                wandb.log(logs, step=state.global_step)
            except Exception as e:
                rank0_print(f"Warning: Failed to log to wandb: {e}")
        
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            """Log evaluation metrics"""
            if wandb.run is None or not metrics:
                return
                
            # Only log from rank 0
            if local_rank != 0 and local_rank != -1:
                return
                
            try:
                # Add eval_ prefix to distinguish from training metrics
                eval_metrics = {f"eval_{k}" if not k.startswith("eval_") else k: v 
                              for k, v in metrics.items()}
                wandb.log(eval_metrics, step=state.global_step)
            except Exception as e:
                rank0_print(f"Warning: Failed to log evaluation metrics to wandb: {e}")
        
        def on_save(self, args, state, control, **kwargs):
            """Log model checkpoints"""
            if wandb.run is None:
                return
                
            # Only log from rank 0
            if local_rank != 0 and local_rank != -1:
                return
                
            if self.log_model in ["all", "end"]:
                try:
                    # Create an artifact for the checkpoint
                    checkpoint_name = f"checkpoint-{state.global_step}"
                    artifact = wandb.Artifact(
                        name=checkpoint_name,
                        type="model",
                        description=f"Model checkpoint at step {state.global_step}, epoch {state.epoch:.2f}"
                    )
                    
                    # Add checkpoint directory to artifact
                    checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)
                    if os.path.exists(checkpoint_dir):
                        artifact.add_dir(checkpoint_dir)
                        wandb.log_artifact(artifact)
                        rank0_print(f"✓ Logged checkpoint {checkpoint_name} to Weights & Biases")
                except Exception as e:
                    rank0_print(f"Warning: Failed to log checkpoint to wandb: {e}")
        
        def on_train_end(self, args, state, control, **kwargs):
            """Final logging and cleanup"""
            if wandb.run is None:
                return
                
            # Only log from rank 0
            if local_rank != 0 and local_rank != -1:
                return
                
            try:
                # Log final model if requested
                if self.log_model == "end":
                    final_artifact = wandb.Artifact(
                        name="final-model",
                        type="model",
                        description=f"Final trained model after {state.global_step} steps"
                    )
                    
                    if os.path.exists(args.output_dir):
                        final_artifact.add_dir(args.output_dir)
                        wandb.log_artifact(final_artifact)
                        rank0_print("✓ Logged final model to Weights & Biases")
                
                # Mark run as finished
                wandb.finish()
                rank0_print("✓ Weights & Biases run completed")
                
            except Exception as e:
                rank0_print(f"Warning: Error during wandb cleanup: {e}")
    
    # Set up callbacks including early stopping if configured
    callbacks = [CUDACacheCleanupCallback(), LossTrackingCallback(training_args.output_dir)]
    
    # Add wandb callback if enabled
    if wandb_config and wandb_config.get("enabled", False):
        callbacks.append(WandbCallback(wandb_config))
    
    # Add early stopping if enabled in config
    early_stopping_patience = training_config.get("early_stopping_patience", None)
    if early_stopping_patience is not None and training_args.eval_strategy != "no":
        rank0_print(f"Setting up early stopping with patience={early_stopping_patience}")
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=training_config.get("early_stopping_threshold", 0.0)
            )
        )
    
    # Prepare the trainer with appropriate callbacks
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module
    )
    
    # Fix for DDP with LoRA: Set static graph to prevent "marked ready twice" error
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        rank0_print("Applying DDP fixes for LoRA compatibility...")
        
        # Method 1: Set static graph on the model
        if hasattr(trainer.model, 'module'):
            # For DDP wrapped models
            if hasattr(trainer.model.module, '_set_static_graph'):
                trainer.model.module._set_static_graph()
                rank0_print("✓ Set static graph on DDP wrapped model")
        elif hasattr(trainer.model, '_set_static_graph'):
            # For models without DDP wrapper
            trainer.model._set_static_graph()
            rank0_print("✓ Set static graph on model")
        
        # Method 2: Ensure all LoRA parameters are properly initialized
        if lora_enable:
            for name, param in trainer.model.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    param.grad = None  # Clear any existing gradients
            rank0_print("✓ Cleared LoRA parameter gradients for clean DDP initialization")
    
    # Check for resume from checkpoint
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Save model state
    trainer.save_state()
    model.config.use_cache = True
    
    # Properly save the model based on training configuration
    if lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), "none"
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            rank0_print(f"Model saved to {training_args.output_dir}")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image-text summarization model using a config file")
    parser.add_argument("--config", type=str, help="Path to the configuration file", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    
    train_image_text_summarization(args.config)