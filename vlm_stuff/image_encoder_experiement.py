from typing import Union, Tuple, Optional

from PIL import Image
import os
import torch
import argparse
import transformers
from transformers import BitsAndBytesConfig
from LLaVA.llava.model import *
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import *
from LLaVA.llava.constants import *
# from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig
from trl import RewardTrainer, PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

"""
Note:
Use transformers-4.31.0 otherwise we get attention mask error...

image_tensor.to(model.device, dtype=torch.float) or we get Half to Float Tensor Error
or just don't use to like in train.py
"""

model_id = "liuhaotian/llava-v1.5-7b"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True,
)

device_map = {
    "transformer.word_embeddings": "cpu",
    "transformer.word_embeddings_layernorm": "cpu",
    "multi_modal_projector": "cpu",
    "vision_tower": "cpu",
    "language_model": "cpu",
    "lm_head": 0,
    "transformer.h": 0,
    "transformer.ln_f": 0,
    "model": 0
}

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
parser.add_argument("--model-base", type=str, default=None)
# parser.add_argument("--image-file", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--conv-mode", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.2)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--load-4bit", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu")
parser.add_argument("--mm_vision_select_layer", type=int, default=2)
parser.add_argument("--mm_use_im_start_end", type=bool, default=False)
parser.add_argument("--mm_use_im_patch_token", type=bool, default=False)
parser.add_argument("--vision_tower", type=str, default="openai/clip-vit-large-patch14-336")
parser.add_argument("--mm_vision_select_feature", type=str, default="patch")
parser.add_argument("--pretrain_mm_mlp_adapter", type=Optional[str], default=None)
parser.add_argument("--tune_mm_mlp_adapter", type=bool, default=True)

args = parser.parse_args()

# 1. Load models
disable_torch_init()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    cache_dir="cache/models",
    model_max_length=200,
    padding_side="right",
    use_fast=False)

config = LlavaConfig.from_pretrained(
    model_id,
    cache_dir="cache/models",
    dtype=torch.float16,

    vocab_size=len(tokenizer),
    n_ctx=1024,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,

    hidden_size=128,
    intermediate_size=688,
    num_hidden_layers=4,
    num_attention_heads=32,
    hidden_act="silu",
    max_position_embeddings=512,
)

model = LlavaLlamaForCausalLM(config).to("cuda")

trl_model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model_name_or_path=model,
                                                              cache_dir="cache/models",
                                                              # peft_config=lora_config,
                                                              ).to("cuda")

model.get_model().initialize_vision_modules(args, fsdp=None)
# model.initialize_vision_tokenizer(args, tokenizer=tokenizer)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    print("Loading vision_tower because it was not loaded.")
    vision_tower.load_model()
vision_tower.to(device="cuda", dtype=torch.float16)

image_processor = vision_tower.image_processor

image = Image.open("img.png").convert('RGB')
image_tensor = process_images([image, image], image_processor, model.config)
print(image_tensor.size())
quit()

prompt = "Hello <image><image>!"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
    model.device)
attention_mask = torch.tensor([[1,1,1,1,1,1]]) # tokenizer(prompt.replace("<image>", "*/_"), return_tensors="pt").get("attention_mask")
print(attention_mask)
model_inputs = model.prepare_inputs_for_generation(input_ids, attention_mask=attention_mask)
print(model_inputs)

(
    input_ids,
    position_ids,
    attention_mask,
    past_key_values,
    inputs_embeds,  # <= this will be used for LlamaModel forward
    labels
            ) = model.prepare_inputs_labels_for_multimodal(input_ids,
                                                              model_inputs['position_ids'],
                                                              model_inputs['attention_mask'],
                                                              model_inputs['past_key_values'],
                                                              labels=input_ids,
                                                              images=image_tensor)
# Note: Each image takes 576 tokens
# In encode_images: CLIP image_features => torch.Size([2, 576, 1024]) => project to => torch.Size([2, 576, 128])
# For each layer we have 1 batch, for each token in context, we have a hidden_size amount of floats
# inputs_embeds has size torch.Size([1, num_of_tokens, hidden_size])
"""
 def encode_images(self, images):
        '''
        We encode images here. Encoding happens in multimodal_encoder, 
        the projector maps from  torch.Size([2, 576, 1024]) to torch.Size([2, 576, 128]) it.
        '''
        image_features = self.get_model().get_vision_tower()(images)
        print("SIZE!!>> ", image_features.size())
        image_features = self.get_model().mm_projector(image_features)
        print("SIZE!!>> ", image_features.size())
        return image_features
"""


# print(input_ids)
print(inputs_embeds, inputs_embeds.size())
# print(labels)
# print(tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
