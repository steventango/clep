import torch
from diffusers import UnCLIPImageVariationPipeline
from diffusers.utils import load_image
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizerFast
from pathlib import Path

def main():
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    vision_model = vision_model.to("cuda")
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)

    urls = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",
        # "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
    ]
    images = [load_image(url) for url in urls]

    image_inputs = processor(images=images, return_tensors="pt", padding=True)
    image_inputs = {k: v.to("cuda") for k, v in image_inputs.items()}

    with torch.no_grad():
        outputs = vision_model(**image_inputs)

    pipe = UnCLIPImageVariationPipeline.from_pretrained("kakaobrain/karlo-v1-alpha-image-variations", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    images = pipe(
        image_embeddings=outputs.image_embeds[:1],
        num_images_per_prompt=5,
    )
    unclip_dir = Path("unclip")
    unclip_dir.mkdir(exist_ok=True)
    for i, image in enumerate(images.images):
        image.save(unclip_dir / f"{i}.png")




if __name__ == "__main__":
    main()
