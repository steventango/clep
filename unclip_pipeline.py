import torch
from diffusers import UnCLIPPipeline
from diffusers.utils import load_image
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizerFast
from pathlib import Path

def main(device):
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16)
    text_model = text_model.to(device)
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

    texts = ["cat"]
    text_inputs = tokenizer(
        texts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with torch.no_grad():
        outputs = text_model(**text_inputs)

    pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
    pipe = pipe.to(device)

    images = pipe(
        text_model_output=outputs,
        text_attention_mask=text_inputs["attention_mask"],
        num_images_per_prompt=5,
    )
    unclip_dir = Path("unclip")
    unclip_dir.mkdir(exist_ok=True)
    for i, image in enumerate(images.images):
        image.save(unclip_dir / f"{i}.png")




if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "mps"
    print(device)
    
    main(device)
