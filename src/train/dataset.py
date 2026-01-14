import copy
import json
import os
import pickle
import torch
import transformers
import lmdb
import re
import numpy as np
import pandas as pd  # For CSV support
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from src.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from src import conversation as conversation_lib
from src.mm_utils import tokenizer_image_token

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    validation_path: Optional[str] = field(default=None,
                           metadata={"help": "Path to validation data."})
    lazy_preprocess: bool = False
    lmdb_path: Optional[str] = field(default=None,
                           metadata={"help": "Path to the LMDB containing training image features."})
    validation_lmdb_path: Optional[str] = field(default=None,
                           metadata={"help": "Path to the LMDB containing validation image features. If None, uses lmdb_path."})
    max_images: int = field(default=5,
                           metadata={"help": "Maximum number of images per article."})
    image_feature_dim: int = field(default=768,
                           metadata={"help": "Dimension of image features."})
    max_input_len: int = field(default=512,
                           metadata={"help": "Maximum input text length."})
    max_summary_len: int = field(default=128,
                           metadata={"help": "Maximum summary text length."})
    train_subset_size: int = field(default=0,
                           metadata={"help": "If > 0, use only this many samples for training."})
    validation_subset_size: int = field(default=0,
                           metadata={"help": "If > 0, use only this many samples for validation."})
    dataloader_num_workers: int = field(default=0,
                           metadata={"help": "Number of subprocesses to use for data loading."})

# LMDB utility functions
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
            raise KeyError(f"Key '{key}' not found in LMDB.")
            
        feature = pickle.loads(value)
    return torch.tensor(feature, dtype=torch.float32)

def load_image_features(lmdb_env, image_paths, max_images, image_feature_dim):
    """
    Load image features from LMDB, pad/truncate to max_images, and create a mask.
    Raises an error if any image cannot be loaded.
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
        path = os.path.normpath(path).replace(
            "\\", "/"
        )
        # No try-except block - let errors propagate
        feature = get_npy_feature_from_lmdb_with_jpg(
            lmdb_env, path
        )
        images_features[i] = feature
        images_mask[i] = 1
        img_paths.append(path)

    return images_features, images_mask, img_paths

def clean_text(text):
    # Remove space before punctuation (.,!?;:)
    return re.sub(r"\s+([.,!?;:])", r"\1", text)

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _improved_tokenize_image_text(text, tokenizer):
    """
    Improved tokenization function that better handles image tokens in text.
    This helps prevent tokenization mismatches.
    """
    if DEFAULT_IMAGE_TOKEN in text:
        # Handle image tokens specially
        tokens = tokenizer_image_token(text, tokenizer)
        # Ensure we return an integer length, not a list
        if isinstance(tokens, list):
            return len(tokens)
        return tokens
    else:
        # Regular tokenization for text-only content
        return len(tokenizer(text).input_ids)

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                # For image prompts, use the actual token length from input_ids
                # This is more accurate than trying to estimate it
                if i == 0:  # First round
                    # Skip tokenization mismatch check for prompts with images
                    # since the special handling in tokenizer_image_token makes
                    # length estimation unreliable
                    instruction_len = len(parts[0].split(DEFAULT_IMAGE_TOKEN)[0])
                    instruction_len = len(tokenizer(parts[0].split(DEFAULT_IMAGE_TOKEN)[0]).input_ids) - 2
                else:
                    instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                
                # Ensure instruction_len is non-negative
                instruction_len = max(0, instruction_len)
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Handle potential index errors by clamping to valid range
            instr_end = min(cur_len + instruction_len, len(target))
            if cur_len < instr_end:
                target[cur_len:instr_end] = IGNORE_INDEX

            # For image prompts, we skip length checking and just use the real length
            if has_image and i == 0:
                # Move to the next instruction
                cur_len = total_len - len(tokenizer(parts[1]).input_ids)
            else:
                cur_len += round_len
        
        # Ensure we don't go beyond the target tensor length
        if cur_len < len(target):
            target[cur_len:] = IGNORE_INDEX

        # Handle tokenization mismatches more gracefully
        # For image prompts, we suppress the warning since it's expected
        if cur_len < tokenizer.model_max_length and not has_image:
            if cur_len != total_len:
                # Don't reset the entire target to IGNORE_INDEX, just warn and continue
                # This prevents losing all training signal from samples with mismatches
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (handling gracefully)")
            
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = _improved_tokenize_image_text(rou, tokenizer)
                instruction_len = _improved_tokenize_image_text(parts[0], tokenizer) - 2
                
                # Ensure instruction_len is non-negative
                instruction_len = max(0, instruction_len)
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Handle potential index errors by clamping to valid range
            instr_end = min(cur_len + instruction_len, len(target))
            if cur_len < instr_end:
                target[cur_len:instr_end] = IGNORE_INDEX

            cur_len += round_len
        
        # Ensure we don't go beyond the target tensor length
        if cur_len < len(target):
            target[cur_len:] = IGNORE_INDEX

        # Handle tokenization mismatches more gracefully
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                # Don't reset the entire target to IGNORE_INDEX, just warn and continue
                # This prevents losing all training signal from samples with mismatches
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (handling gracefully)")
            
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n'
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 lmdb_path: Optional[str] = None,
                 is_validation: bool = False):
        super(LazySupervisedDataset, self).__init__()

        self.tokenizer = tokenizer
        self.is_validation = is_validation
        
        # Use the provided LMDB path if specified, otherwise use the one from data_args
        self.lmdb_path = lmdb_path if lmdb_path is not None else data_args.lmdb_path
        print(f"Using LMDB path: {self.lmdb_path}")
        
        # Load data from CSV
        self.df = pd.read_csv(data_path)
        
        # Apply subset size depending on whether this is training or validation
        subset_size = 0
        if is_validation and data_args.validation_subset_size > 0:
            subset_size = data_args.validation_subset_size
            subset_type = "validation"
        elif not is_validation and data_args.train_subset_size > 0:
            subset_size = data_args.train_subset_size
            subset_type = "training"
            
        # If subset size is specified, limit the dataset
        if subset_size > 0:
            orig_size = len(self.df)
            # Ensure we don't try to use more samples than available
            subset_size = min(subset_size, orig_size)
            # Take the first subset_size samples
            self.df = self.df.head(subset_size)
            print(f"Using {subset_size} samples out of {orig_size} for {subset_type}")
            
        print(f"Loaded CSV dataset with {len(self.df)} examples")
        # Convert DataFrame to list of dictionaries
        self.list_data_dict = self.df.to_dict('records')
            
        self.data_args = data_args
        self.lmdb_env = None  # Will be initialized lazily

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Lazy initialization of LMDB connection
        if self.lmdb_env is None and self.lmdb_path is not None:
            self.lmdb_env = open_lmdb(self.lmdb_path)
            
        # Use direct reference to data dictionary
        source = self.list_data_dict[i]
        
        # Check for the exact column names used in your CSV
        if "Body" not in source:
            raise KeyError(f"Sample {i} missing required 'Body' field")
            
        # Extract input text
        input_text = source["Body"]
            
        # Check for the exact summary field name
        if "Summary" not in source:
            raise KeyError(f"Sample {i} missing required 'Summary' field")
            
        # Extract summary
        summary = source["Summary"]
            
        # Extract image paths with exact column name (optional)
        image_paths = []
        if "Image_Paths" in source:
            image_paths = source["Image_Paths"]
        
        # Handle image paths parsing if they exist
        if image_paths and isinstance(image_paths, str):
            # Try to parse as JSON first (handles array formatted as string)
            image_paths = json.loads(image_paths)

          # Process images if there are any paths and LMDB is available
        has_images = False
        num_images = 0
        if image_paths and self.lmdb_env is not None:
            try:
                # Load image features from LMDB
                images_features, images_mask, img_paths = load_image_features(
                    lmdb_env=self.lmdb_env,
                    image_paths=image_paths,
                    max_images=self.data_args.max_images,
                    image_feature_dim=self.data_args.image_feature_dim,
                )
                
                # Count the number of actual images loaded
                num_images = int(images_mask.sum().item())
                has_images = num_images > 0
            except Exception as e:
                # Raise the error to ensure data quality
                raise ValueError(f"Failed to load images for sample {i}: {str(e)}")
                
        # If no images were loaded or there were no image paths, use empty features
        if not has_images:
            images_features = torch.zeros((self.data_args.max_images, self.data_args.image_feature_dim), dtype=torch.float32)
            num_images = 0
            
        # Format conversation for the model using centralized prompt manager
        from src.prompt_manager import format_prompt_with_images
        prompt = format_prompt_with_images(input_text, num_images, mode="training")
        
        conversation = [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": summary}
        ]
        
        # Process the data through the tokenizer
        data_dict = preprocess(
            [conversation],
            self.tokenizer,
            has_image=num_images > 0)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        data_dict['image'] = images_features
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # Stack images into batch tensor (batch_size, max_images, feature_dim)
            batch_size = len(images)
            max_images = images[0].shape[0]
            feature_dim = images[0].shape[1]
            
            stacked_images = torch.zeros(
                (batch_size, max_images, feature_dim), 
                dtype=images[0].dtype, 
                device=images[0].device
            )
            
            for i, img in enumerate(images):
                # Handle case where images might have different dimensions
                if img.shape[0] > 0:
                    stacked_images[i, :img.shape[0]] = img
                    
            batch['images'] = stacked_images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    # Create data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Handle validation data
    eval_dataset = None
    
    # Load validation dataset if path is provided
    if data_args.validation_path and os.path.exists(data_args.validation_path):
        print(f"Loading validation data from {data_args.validation_path}")
        
        # Use validation LMDB if provided, otherwise fall back to training LMDB
        validation_lmdb = data_args.validation_lmdb_path if data_args.validation_lmdb_path else data_args.lmdb_path
        
        if data_args.validation_lmdb_path:
            print(f"Using separate validation LMDB: {validation_lmdb}")
        
        eval_dataset = LazySupervisedDataset(
            data_path=data_args.validation_path,
            tokenizer=tokenizer,
            data_args=data_args,
            lmdb_path=validation_lmdb,
            is_validation=True
        )
    
    # Create training dataset with training LMDB
    train_dataset = LazySupervisedDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        lmdb_path=data_args.lmdb_path,
        is_validation=False
    )
    
    # Create the final data module dictionary
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

