import datetime
import logging
import logging.handlers
import os
import sys

import requests
import torch
import hashlib
import subprocess
import urllib.request

from src.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    """
    A logger that writes to both standard output and file for the distributed setting.
    Each rank only has one logger.
    """
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if logger_filename is not None:
        handler = logging.FileHandler(logger_filename)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def get_sequence_length_ratio(conversation):
    # WARNING: this function requires src to be imported.
    return float(len(conversation.get_prompt()) / (conversation.offset + 1))


def get_sorted_list(conversations):
    # Calculate the ratio for sorting
    length_ratio = [get_sequence_length_ratio(conversation) for conversation in conversations]
    
    # Create a list of (ratio, index) tuples
    indexed_ratios = [(ratio, idx) for idx, ratio in enumerate(length_ratio)]
    
    # Sort by ratio
    sorted_ratios = sorted(indexed_ratios, key=lambda x: x[0])
    
    # Extract the sorted indices
    sorted_indices = [idx for ratio, idx in sorted_ratios]
    
    return sorted_indices


def detect_flash_attention_support():
    """
    Detect if Flash Attention 2 is available and supported by the current GPU.
    
    Returns:
        str: "flash_attention_2" if supported, "sdpa" if CUDA available but no Flash Attention, 
             "eager" if no CUDA
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        return "eager"
    
    # Check if Flash Attention package is installed and functional
    try:
        import flash_attn
        # Try to actually use a Flash Attention function to ensure it works
        # This will catch GLIBC and other runtime issues
        from flash_attn import flash_attn_func
        flash_attn_available = True
    except ImportError as e:
        print(f"Flash Attention not available: {e}")
        flash_attn_available = False
    except Exception as e:
        # Catch runtime errors like GLIBC version mismatches
        print(f"Flash Attention import failed: {e}")
        if "GLIBC" in str(e):
            print("GLIBC version compatibility issue detected. Falling back to SDPA.")
            print("See docs/FLASH_ATTENTION_SETUP.md for installation alternatives.")
        flash_attn_available = False
    
    if not flash_attn_available:
        # Use SDPA as fallback for CUDA devices
        return "sdpa"
    
    # Check GPU capability for Flash Attention
    try:
        cuda_major, cuda_minor = torch.cuda.get_device_capability()
        
        # Flash Attention 2 requires compute capability >= 8.0 (Ampere and newer)
        # But also works on some 7.5 (Turing) GPUs
        if cuda_major >= 8 or (cuda_major == 7 and cuda_minor >= 5):
            return "flash_attention_2"
        else:
            # GPU doesn't support Flash Attention, use SDPA
            print(f"GPU compute capability {cuda_major}.{cuda_minor} doesn't support Flash Attention. Using SDPA.")
            return "sdpa"
            
    except Exception as e:
        # If we can't determine GPU capability, fallback to SDPA
        print(f"Error determining GPU capability: {e}")
        return "sdpa"


def get_attention_implementation(prefer_flash_attention=True):
    """
    Get the best available attention implementation.
    
    Args:
        prefer_flash_attention (bool): Whether to prefer Flash Attention if available
        
    Returns:
        str: The attention implementation to use
    """
    if prefer_flash_attention:
        implementation = detect_flash_attention_support()
    else:
        # Force fallback to SDPA or eager
        if torch.cuda.is_available():
            implementation = "sdpa"
        else:
            implementation = "eager"
    
    return implementation


def print_attention_info(implementation):
    """Print information about the attention implementation being used."""
    if implementation == "flash_attention_2":
        try:
            import flash_attn
            version = flash_attn.__version__
            cuda_major, cuda_minor = torch.cuda.get_device_capability()
            print(f"Using Flash Attention 2 (v{version}) on GPU with compute capability {cuda_major}.{cuda_minor}")
        except:
            print("Using Flash Attention 2")
    elif implementation == "sdpa":
        print("Using PyTorch native scaled_dot_product_attention (SDPA)")
    elif implementation == "eager":
        print("Using standard (eager) attention implementation")
    else:
        print(f"Using attention implementation: {implementation}")
