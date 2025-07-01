import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

img_path = "lalaland_text.png"
raw_image = Image.open(img_path).convert('RGB')

# conditional image captioning
text = "An image of "
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")


# unconditional image captioning
# inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(out[0], skip_special_tokens=True))
