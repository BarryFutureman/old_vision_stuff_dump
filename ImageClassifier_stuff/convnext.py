from transformers import ConvNextImageProcessor, ConvNextForImageClassification
import torch
from datasets import load_dataset
from PIL import Image

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]
img_path = "lalaland_text.png"
raw_image = Image.open(img_path).convert('RGB')

processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-large-224")
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224")

inputs = processor(raw_image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])