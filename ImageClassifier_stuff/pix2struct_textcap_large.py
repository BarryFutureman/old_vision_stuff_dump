import torch

from PIL import Image
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

img_path = "chrome_img43.jpg"
raw_image = Image.open(img_path).convert('RGB')

model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-large")
processor = Pix2StructProcessor.from_pretrained("google/pix2struct-textcaps-large")

# image only
inputs = processor(images=raw_image, return_tensors="pt")

predictions = model.generate(**inputs)
print(processor.decode(predictions[0], skip_special_tokens=True))
