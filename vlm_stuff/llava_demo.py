from PIL import Image
import torch
from transformers import BitsAndBytesConfig
from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig
from transformers import AutoProcessor, LlavaProcessor

image = Image.open("video_to_image_outputs/sample/frame_0000.jpg")
image1 = Image.open("video_to_image_outputs/sample/frame_0004.jpg")
image2 = Image.open("video_to_image_outputs/sample/frame_0009.jpg")
image3 = Image.open("video_to_image_outputs/sample/frame_0010.jpg")


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

'''
# Initializing a CLIP-vision config
vision_config = CLIPVisionConfig()

# Initializing a Llama config
text_config = LlamaConfig()

# Initializing a Llava llava-1.5-7b style configuration
configuration = LlavaConfig(vision_config, text_config)'''

# Initializing a model from the llava-1.5-7b style configuration
model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                      cache_dir="cache/models",
                                                      quantization_config=quantization_config,
                                                      device_map=device_map)

processor = LlavaProcessor.from_pretrained(model_id, cache_dir="cache/models",)

prompt = "<image><image><image><image>\nUSER: Describe the man's action in the sequence of images.\nASSISTANT:"

inputs = processor(text=prompt, images=[image, image1, image2, image3], return_tensors="pt")  # .to("cuda")

# Generate
generate_ids = model.generate(**inputs, max_length=200)  # .generate is from GenerationMixin
# prepare_inputs_for_generation is an abstract function in GenerationMixin but implemented in modeling_llava.py

result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(result)
