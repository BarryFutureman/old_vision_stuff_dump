import torch
from transformers import pipeline
from PIL import Image

model_id = "llava-hf/bakLlava-v1-hf"
pipe = pipeline("image-to-text", model=model_id, torch_dtype=torch.float16, device_map="auto")

image = Image.open("output_image.png")
image2 = Image.open("img.png")
prompt = "USER: <image><image2>\nWhat the fuck are these?\nASSISTANT:"

outputs = pipe([image, image2], prompt=prompt, generate_kwargs={"max_new_tokens": 200})
print(outputs)

'''
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/bakLlava-v1-hf"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
)

processor = AutoProcessor.from_pretrained(model_id)


image = Image.open("output_image.png")
prompt = "USER: <image>\nWhat the fuck is this?\nASSISTANT:"
inputs = processor(prompt, image, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))'''

