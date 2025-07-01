from diffusers import StableDiffusionXLImg2ImgPipeline, AutoPipelineForText2Image, StableDiffusionXLPipeline
import torch
from diffusers.utils import load_image, make_image_grid
from PIL import Image

# use from_pipe to avoid consuming additional memory when loading a checkpoint
pipeline = StableDiffusionXLImg2ImgPipeline.from_single_file("models/dreamshaperXL_turboDpmppSDE.safetensors",
                                                             local_files_only=False,
                                                             torch_dtype=torch.float16)
# pipeline.to("cuda")
# pipeline.enable_vae_slicing()
# pipeline.enable_vae_tiling()
pipeline.enable_sequential_cpu_offload()
# pipeline.enable_model_cpu_offload()
# pipeline.to("cuda")

init_image = load_image("img.png")
init_image = init_image.resize((512, 512))

prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"

image = pipeline(prompt, image=init_image, strength=0.5, guidance_scale=0.1, num_inference_steps=8).images[0]
img_result = make_image_grid([init_image, image], rows=1, cols=2)

img_result.save("generated.png")

