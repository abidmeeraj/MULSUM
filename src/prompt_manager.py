"""
Centralized Prompt Management for Multimodal Summarization

This module provides a unified interface for managing prompts across training and inference,
optimized for better ROUGE scores and consistent multimodal summarization performance.
"""

import json
import os
from typing import Dict, Any, Optional

class PromptManager:
    """Manages prompts for multimodal summarization with ROUGE optimization."""
    
    def __init__(self, prompts_config_path: str = "configs/prompts_config.json"):
        """
        Initialize the prompt manager.
        
        Args:
            prompts_config_path: Path to the prompts configuration file
        """
        self.config_path = prompts_config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the prompts configuration from file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Prompts config file not found: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    

    
    def get_system_message(self, template_name: str = "multimodal_summarization") -> str:
        """
        Get the system message for conversation templates.
        
        Args:
            template_name: Name of the conversation template
            
        Returns:
            System message string
        """
        return self.config["conversation_templates"][template_name]["system_message"]
    
    def get_prompt(self, has_images: bool, mode: str = "inference", text_content: str = "") -> str:
        """
        Get the appropriate prompt for the given context.
        
        Args:
            has_images: Whether images are available
            mode: 'training' or 'inference'
            text_content: The text content to be summarized
            
        Returns:
            Formatted prompt string
        """
        prompt_type = "with_images" if has_images else "without_images"
        prompt_key = f"{mode}_prompt"
        
        template = self.config["summarization_prompts"][prompt_type][prompt_key]
        return template.format(text_content=text_content)
    
    def get_training_prompt(self, has_images: bool, text_content: str) -> str:
        """Get training prompt."""
        return self.get_prompt(has_images, "training", text_content)
    
    def get_inference_prompt(self, has_images: bool, text_content: str) -> str:
        """Get inference prompt."""
        return self.get_prompt(has_images, "inference", text_content)
    
    def get_prompt_settings(self) -> Dict[str, Any]:
        """Get prompt settings for configuration."""
        return self.config.get("prompt_settings", {})
    
    def get_rouge_optimization_info(self) -> Dict[str, Any]:
        """Get ROUGE optimization information."""
        return self.config.get("rouge_optimization", {})
    
    def format_prompt_with_images(self, text_content: str, num_images: int, mode: str = "inference") -> str:
        """
        Format prompt with appropriate image tokens.
        
        Args:
            text_content: The text content to summarize
            num_images: Number of images available
            mode: 'training' or 'inference'
            
        Returns:
            Formatted prompt with image tokens
        """
        has_images = num_images > 0
        base_prompt = self.get_prompt(has_images, mode, text_content)
        
        if has_images:
            # Use multiple image tokens for both training and inference to ensure consistency
            from src.constants import DEFAULT_IMAGE_TOKEN
            image_tokens = " ".join([DEFAULT_IMAGE_TOKEN] * num_images)
            return f"{image_tokens}\n{base_prompt}"
        else:
            return base_prompt

# Global instance for easy access
_prompt_manager = None

def get_prompt_manager(config_path: Optional[str] = None) -> PromptManager:
    """
    Get the global prompt manager instance.
    
    Args:
        config_path: Optional path to prompts config file
        
    Returns:
        PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None or config_path is not None:
        _prompt_manager = PromptManager(config_path or "configs/prompts_config.json")
    return _prompt_manager

def reload_prompts(config_path: Optional[str] = None):
    """Reload prompts configuration."""
    global _prompt_manager
    _prompt_manager = PromptManager(config_path or "configs/prompts_config.json")

# Convenience functions
def get_system_message(template_name: str = "multimodal_summarization") -> str:
    """Get system message for conversation template."""
    return get_prompt_manager().get_system_message(template_name)

def get_training_prompt(has_images: bool, text_content: str) -> str:
    """Get training prompt."""
    return get_prompt_manager().get_training_prompt(has_images, text_content)

def get_inference_prompt(has_images: bool, text_content: str) -> str:
    """Get inference prompt."""
    return get_prompt_manager().get_inference_prompt(has_images, text_content)

def format_prompt_with_images(text_content: str, num_images: int, mode: str = "inference") -> str:
    """Format prompt with image tokens."""
    return get_prompt_manager().format_prompt_with_images(text_content, num_images, mode) 