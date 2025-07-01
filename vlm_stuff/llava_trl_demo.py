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

    hidden_size=256,
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
# mm_projector_weights = torch.load(os.path.join("cache/models/projector", 'mm_projector.bin'), map_location='cpu')
# mm_projector_weights = {k: v.to(torch.half) for k, v in mm_projector_weights.items()}
# model.load_state_dict(mm_projector_weights, strict=False)

model.get_model().initialize_vision_modules(args, fsdp=None)
# model.initialize_vision_tokenizer(args, tokenizer=tokenizer)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    print("Loading vision_tower because it was not loaded.")
    vision_tower.load_model()
vision_tower.to(device="cuda", dtype=torch.float16)

image_processor = vision_tower.image_processor

image = Image.open("img.png").convert('RGB')
image_tensor = process_images([image], image_processor, model.config)
'''if type(image_tensor) is list:
    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
else:
    image_tensor = image_tensor.to(model.device, dtype=torch.float)'''

"""prompt = "<USER> What color is this <image>?\n<ASSISTANT>"
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(
    model.device)
keywords = ["\n"]
stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
streamer = transformers.TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)


with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=200,
        streamer=streamer,
        use_cache=True,
        stopping_criteria=[stopping_criteria])

result = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
print(result)"""


# Create Trainer
ppo_config = PPOConfig(
    model_name="outputs/ColorModel",
    learning_rate=0.001,
    batch_size=1,  # Set batch size or dataloader will not work
    ratio_threshold=20
)

ppo_trainer = PPOTrainer(
    model=trl_model,
    config=ppo_config,
    dataset=None,
    data_collator=None,
    tokenizer=tokenizer
)


# Start Training
generation_kwargs = {
    "images": image_tensor,
    "min_length": -1,
    "temperature": 1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 2,
}

for i in range(5):
    prompt = "<USER> What color is this <image>?\n<ASSISTANT>"
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

    image = Image.open("img.png").convert('RGB')
    image_tensor = process_images([image], image_processor, model.config)

    query_tensors = list(input_ids)
    response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
    print(response_tensors)
    positive_tensors = response_tensors[0][response_tensors[0] >= 0]
    response_text = str(tokenizer.decode(positive_tensors))
    print(response_text)

    reward = 1 if "red" in response_text.lower() else 0
    rewards = [torch.tensor(float(reward))]

    stats = ppo_trainer.step([input_ids[input_ids >= 0]], response_tensors, rewards)
    print(stats)
    # ppo_trainer.log_stats(stats, batch, rewards)

