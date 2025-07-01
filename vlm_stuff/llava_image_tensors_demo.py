from PIL import Image
import torch
from transformers import BitsAndBytesConfig
from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig
from transformers import AutoProcessor, LlavaProcessor, CLIPImageProcessor

image = Image.open("img.png")
image1 = Image.open("img_1.png")


model_id = "llava-hf/llava-1.5-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True,
)

device_map = {
    "transformer.word_embeddings": "cpu",
    "transformer.word_embeddings_layernorm": "cpu",
    "multi_modal_projector": "cpu",
    "vision_tower": 0,
    "language_model": "cpu",
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}

vision_config = CLIPVisionConfig()


# Initializing a model from the llava-1.5-7b style configuration
model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                      cache_dir="cache/models",
                                                      quantization_config=quantization_config,
                                                      device_map=device_map)

processor = LlavaProcessor.from_pretrained(model_id, cache_dir="cache/models",)

prompt = "<image>\nand then turn into\n<image>\nUSER: The color changed from what to what?\nASSISTANT:"

# inputs = processor(text=prompt, images=[image], return_tensors="pt")  # .to("cuda")
inputs2 = processor(text=prompt, images=[image, image1], return_tensors="pt").to("cuda")

# print(inputs["pixel_values"].size())
print(inputs2["pixel_values"].size())

# Generate
generate_ids = model.generate(**inputs2, max_length=200)
result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(result)
