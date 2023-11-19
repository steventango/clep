

from datasets import load_dataset
from transformers import CLIPVisionModelWithProjection

from transformers import CLIPTokenizerFast

preprocessed_dataset = load_dataset("eeg_mnist_preprocessed", keep_in_memory=True)

model = CLIPVisionModelWithProjection.from_pretrained("eeg_mnist_preprocessed")

map_digit_to_token = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

map_digit_to_input_ids = {
    digit: tokenizer(f"{map_digit_to_token[digit]}", padding=True).input_ids
    for digit in range(10)
}

import torch

inputs = {
    "pixel_values": processor(images=images, return_tensors="pt").pixel_values,
    "input_ids": torch.tensor([map_digit_to_input_ids[digit] for digit in range(10)]),
    "attention_mask": torch.tensor([[1] * 3] * 10),
}
outputs = model(**inputs, return_loss=True)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
