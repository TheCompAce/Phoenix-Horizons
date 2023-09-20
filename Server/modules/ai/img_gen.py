# https://huggingface.co/warp-ai/wuerstchen
import argparse
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
import numpy as np
import os


device = "cuda"
dtype = torch.float16

pipeline = AutoPipelineForText2Image.from_pretrained(
    "warp-diffusion/wuerstchen", torch_dtype=dtype
).to(device)

def generate_image(prompt, width=1024, height=1024, prior_guidance_scale=4.0, decoder_guidance_scale=0.0, num_variations=1):
    outputs = []
    for _ in range(num_variations):
        output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            prior_guidance_scale=prior_guidance_scale,
            decoder_guidance_scale=decoder_guidance_scale,
        ).images
        outputs.extend(output)
    return outputs

def save_image(imgs, filename, width, height):
    base_filename, extension = os.path.splitext(filename)
    
    for idx, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            image_array = img.permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray((image_array * 255).astype('uint8'))
        else:
            image = img

        image = image.resize((width, height), Image.ANTIALIAS)
        
        if idx == 0:
            image.save(filename)
        else:
            image.save(f"{base_filename}_{idx}{extension}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from text prompts.')
    parser.add_argument('prompt', type=str, help='The text prompt to generate images from.')
    parser.add_argument('output_filename', type=str, help='The filename to save the generated image(s) as.')
    parser.add_argument('--num_variations', type=int, default=1, help='The number of image variations to generate.')

    args = parser.parse_args()

    if os.path.isfile(args.prompt):
        with open(args.prompt, 'r') as file:
            caption = file.read()
    else:
        caption = args.prompt

    caption = args.prompt
    output_filename = args.output_filename
    num_variations = args.num_variations
    
    print(f"Generating {num_variations} image(s) from prompt: {caption}")
    image_output = generate_image(caption, num_variations=num_variations)
    save_image(image_output, output_filename, 1024, 1024)

