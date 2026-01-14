from PIL import Image
from io import BytesIO
import base64
import torch
from transformers import StoppingCriteria
from src.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# Note: KeywordsStoppingCriteria has been updated to support beam search (2025-06-25)
# Previously, the class only supported batch_size=1, which caused issues when using beam search
# Now it can handle multiple beams (num_beams > 1) by checking each sequence in the batch


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Modified to support beam search (num_beams > 1)
        # Instead of asserting batch_size==1, we'll handle it appropriately
        batch_size = output_ids.shape[0]
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        
        # Check if any sequence in the batch contains a stopping keyword
        for b in range(batch_size):
            # Check token IDs
            for keyword_id in self.keyword_ids:
                if output_ids[b, -keyword_id.shape[0]:].equal(keyword_id):
                    return True
            
            # Check decoded text for keywords
            output_text = self.tokenizer.decode(output_ids[b, -offset:], skip_special_tokens=True)
            for keyword in self.keywords:
                if keyword in output_text:
                    return True
        
        return False


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )