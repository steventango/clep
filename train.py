import itertools
import logging

import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    CLIPProcessor,
    CLIPTokenizerFast,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    Trainer,
    TrainingArguments,
)
from transformers.utils.logging import set_verbosity_warning

from trainable_clip_model import TrainableCLIPModel
from wavelet_spectrogram import cwt_spectrogram

set_verbosity_warning()
logging.basicConfig(filename='train.log', level=logging.INFO)

dataset = load_dataset("DavidVivancos/MindBigData2022_MNIST_IN")
logging.info(list(dataset["train"][0].items())[:8])

# %%
channels = sorted(set(map(lambda x: x.split("-")[0], dataset["train"][0].keys())) - {'label'})
logging.info(channels)


# %%
# B, C, H, W
C = len(channels)
H, W = 168, 256
L = 256
S = max(H, W)


# %%
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


# %%
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")


# %%
def preprocess(inputs, return_loss=True):
  labels = [map_digit_to_token[label] for label in inputs["label"]]
  tokenized_labels = tokenizer(labels, padding=True, return_tensors="pt")
  inputs["input_ids"] = tokenized_labels.input_ids
  inputs["attention_mask"] = tokenized_labels.attention_mask
  B = len(inputs["label"])
  samples = np.zeros((B, C, S, S))
  for b in range(B):
    for i, channel in enumerate(channels):
      sample = np.array([inputs[f"{channel}-{j}"][b] for j in range(L)])
      power, *_ = cwt_spectrogram(sample, 120, nNotes=24, detrend=True, normalize=True)
      samples[b, i, :min(H, S), :min(W, S)] = power.squeeze()
  inputs["pixel_values"] = samples
  return inputs


# %%
item = preprocess(dataset["train"][:8])
item["label"], item["input_ids"], item['pixel_values'].shape


# %%
remove_columns = [f"{channel}-{i}" for channel, i in itertools.product(channels, range(L))] + ["label"]


# %%
preprocessed_dataset = dataset.map(preprocess, batched=True, remove_columns=remove_columns, num_proc=24)


# %%
configuration = CLIPVisionConfig(
    image_size=max(H, W),
    num_channels=C,
    num_hidden_layers=1,
    # hidden_size (`int`, *optional*, defaults to 768):
    #     Dimensionality of the encoder layers and the pooler layer.
    hidden_size=128,
    # intermediate_size (`int`, *optional*, defaults to 3072):
    #     Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    intermediate_size=4*128,
    # projection_dim (`int`, *optional*, defaults to 512):
    #     Dimentionality of text and vision projection layers.
    # num_hidden_layers (`int`, *optional*, defaults to 12):
    #     Number of hidden layers in the Transformer encoder.
    # num_attention_heads (`int`, *optional*, defaults to 12):
    #     Number of attention heads for each attention layer in the Transformer encoder.
    num_attention_heads=4,
)
vision_model = CLIPVisionModelWithProjection(configuration)


# %%
model = TrainableCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# %%
model.vision_model = vision_model.vision_model


# %%
for p in model.text_model.parameters():
  p.requires_grad = False
for p in model.text_projection.parameters():
  p.requires_grad = False


# %%
logging.info(model)
logging.info(f"All Parameters: {sum(p.numel() for p in model.parameters()) / 1e6}M")
logging.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")


# %%
preprocessed_dataset['train'], preprocessed_dataset['validation'] = preprocessed_dataset['train'].train_test_split(.1).values()


# %%
training_args = TrainingArguments(
    output_dir="clep_trainer",
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    logging_steps=1,
    tf32=True,
    num_train_epochs=5,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    weight_decay=0.1,
    warmup_steps=1_000,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    torch_compile=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed_dataset["train"],
    eval_dataset=preprocessed_dataset["validation"],
    # compute_metrics=compute_metrics,
)


# %%
trainer.train()


# %%
