# %%
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "stabilityai/stable-diffusion-2-1-base"
controlnet_path = "model_out"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed
pipe.enable_xformers_memory_efficient_attention()

pipe.enable_model_cpu_offload()

control_image = load_image("./flickr8k/train_img.png")
prompt = "the boy laying face down on skateboard is being pushed along the ground by another boy"

# generate image
generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]

image.save("./output.png")
# %%
# show output.png and val_img.png
import matplotlib.pyplot as plt
import PIL
im = PIL.Image.open("./output.png")
fig, ax = plt.subplots(1, 2)

ax[0].imshow(PIL.Image.open("./flickr8k/val_img.png"))
ax[0].set_title("input")
ax[1].imshow(im)
ax[1].set_title("output")
# display prompt
fig.text(0.5, 0.05, prompt, ha='center')
plt.show()
# %%
