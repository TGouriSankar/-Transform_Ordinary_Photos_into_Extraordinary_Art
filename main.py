import torch
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionPipeline

# Use a pipeline as a high-level helper
from transformers import pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

caption_image = pipeline("image-to-text",
                model="Salesforce/blip-image-captioning-large", device=device)


def image_generation(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    #pipeline.to(device)
    pipeline.enable_model_cpu_offload()
    
    image = pipeline(
        prompt=prompt + " 8K, Ultra HD",
        negative_prompt="blurred, ugly, watermark, low resolution, blurry, nude",
        num_inference_steps=40,
        height=1024,
        width=1024,
        guidance_scale=9.0
    ).images[0]

    return image

def caption_my_image(pil_image):
    semantics = caption_image(images=pil_image)[0]['generated_text']
    images = image_generation(semantics)
    return images

demo = gr.Interface(fn=caption_my_image,
                    inputs=[gr.Image(label="Select Image",type="pil")],
                    outputs=[gr.Image(label="New Image genrated using SD3",type="pil")],
                    title="PicTalker | ImageNarrator | SnapSpeech | SpeakScene",
                    description="🌟 Transform Ordinary Photos into Extraordinary Art!")
demo.launch()