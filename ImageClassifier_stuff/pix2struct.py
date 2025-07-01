from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import torch

img_path = "webpage.png"
raw_image = Image.open(img_path).convert('RGB')

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-widget-captioning-base", torch_dtype=torch.bfloat16).to("cuda")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-widget-captioning-base")

text = "The page is"
inputs = processor(images=raw_image, text=text, return_tensors="pt").to("cuda", torch.bfloat16)

predictions = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(predictions[0], skip_special_tokens=True))