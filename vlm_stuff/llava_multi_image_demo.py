from PIL import Image
from io import BytesIO
import os
import torch
from transformers import BitsAndBytesConfig
from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig
from transformers import AutoProcessor, LlavaProcessor, CLIPImageProcessor
import argparse
'''from LLaVA.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from LLaVA.llava.conversation import conv_templates, SeparatorStyle
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria'''
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        import requests
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def concatenate_images(images: list[Image]):
    width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + 20 * (len(images) - 1)

    new_img = Image.new('RGB', (width, total_height), (0, 0, 0))

    current_height = 0
    for img in images:
        new_img.paste(img, (0, current_height))
        current_height += img.height + 20  # adding a 20px black bar

    new_img.save("concatenated_image.png")

    return new_img


def images_in_folder_to_pil_list(folder_path):
    pil_image_list = []

    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Filter out non-image files (you may want to adjust this based on your file types)
    image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Sort the files for consistency (optional)
    image_files.sort()

    # Load each image and append it to the list
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        pil_image = Image.open(image_path)
        pil_image_list.append(pil_image)

    return pil_image_list


if __name__ == "__main__":
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
        "vision_tower": "cpu",
        "language_model": "cpu",
        "lm_head": "cpu",
        "transformer.h": "cpu",
        "transformer.ln_f": "cpu",
    }

    # Initializing a model from the llava-1.5-7b style configuration
    # seed_value = 4288
    # torch.manual_seed(seed_value)

    model = LlavaForConditionalGeneration.from_pretrained(model_id,
                                                          cache_dir="cache/models",
                                                          quantization_config=quantization_config,
                                                          device_map=device_map
                                                          )
    print("Torch Seed:", torch.initial_seed())  # 134512244632300
    print(model.multi_modal_projector.linear_2.weight.cpu())
    quit()

    processor = AutoProcessor.from_pretrained(model_id, cache_dir="cache/models", )

    image_lst = images_in_folder_to_pil_list("video_to_image_outputs/vid")

    image_tokens = "\n".join(["<image>" for i in image_lst])
    prompt = f"The following is a video as an Image Sequence:\n{image_tokens}\nUSER: What did the man do in the end of the video?\nASSISTANT:"
    print("\"\"\"\n", prompt, "\n\"\"\"")

    inputs = processor(text=prompt, images=image_lst, return_tensors="pt").to("cuda")

    # print(inputs["pixel_values"].size())
    print(inputs["pixel_values"].size())
    print(processor.batch_decode(inputs["input_ids"]))

    # Generate
    generate_ids = model.generate(**inputs, max_length=200)
    result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(result)
