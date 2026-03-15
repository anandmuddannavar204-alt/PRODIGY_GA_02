from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

# Prompt for image generation
prompt = "A futuristic cyberpunk city at night with neon lights"

# Generate image
image = pipe(prompt).images[0]

print("Prompt:", prompt)

# Save image
image.save("generated_image.png")

# Show image
image
