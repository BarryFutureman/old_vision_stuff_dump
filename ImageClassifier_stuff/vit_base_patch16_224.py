from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import requests

image = Image.open("elephant_flat.png")
# Convert the image to RGB mode (if necessary)
if image.mode != 'RGB':
    image = image.convert('RGB')

# Convert the image to 24-bit depth
converted_image = image.convert('RGB')
image = converted_image

# converted_image.save("teapot_converted.png")

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

inputs = processor(images=image, return_tensors="pt")
inputs = inputs.to(device)
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
