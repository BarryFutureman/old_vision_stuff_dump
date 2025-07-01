from diffusers import LCMScheduler, AutoPipelineForText2Image
import torch


class LCMSDInfer:
    def __init__(self):
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.adapter_id = "latent-consistency/lcm-lora-sdv1-5"

        self.pipe = AutoPipelineForText2Image.from_pretrained(pretrained_model_or_path=self.model_id,
                                                              cache_dir="cache",
                                                              # safety_checker=None,
                                                              torch_dtype=torch.float16,
                                                              variant="fp16",
                                                              )
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")

        # load and fuse lcm lora
        self.pipe.load_lora_weights(self.adapter_id)
        self.pipe.fuse_lora()

        self.width = 512
        self.height = 512
        self.steps = 4
        self.guidance_scale = 0.8

    def set_params(self, w, h, s, g):
        self.width = w
        self.height = h
        self.steps = s
        self.guidance_scale = g

    def predict(self, prompt: str):
        image = self.pipe(prompt, width=self.width, height=self.height,
                          num_inference_steps=4, guidance_scale=0.8).images[0]

        output_file = "output_image.png"
        image.save(output_file)

        return output_file
