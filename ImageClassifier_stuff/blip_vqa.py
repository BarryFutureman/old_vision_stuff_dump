import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to("cuda")

img_path = "lalaland_text.png"
raw_image = Image.open(img_path).convert('RGB')

question = "Is she watching a movie?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

out = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(out[0], skip_special_tokens=True))