import itertools
import logging

import numpy as np
from datasets import load_dataset
from transformers import CLIPTokenizerFast
from transformers.utils.logging import set_verbosity_warning

from trainable_clip_model import TrainableCLIPModel
from wavelet_spectrogram import cwt_spectrogram

set_verbosity_warning()
logging.basicConfig(filename='train.log', level=logging.INFO)

dataset = load_dataset("DavidVivancos/MindBigData2022_MNIST_MU")
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
    -1: "none",
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

map_digit_to_input_ids = {
    digit: tokenizer(f"{map_digit_to_token[digit]}", padding=True).input_ids
    for digit in range(-1, 10)
}

def preprocess(inputs):
  B = len(inputs["label"])
  inputs["labels"] = inputs["label"]
  inputs["input_ids"] = [map_digit_to_input_ids[label] for label in inputs["label"]]
  inputs["attention_mask"] = [[1] * 3] * B
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
item["labels"], item["input_ids"], item['pixel_values'].shape


# %%
remove_columns = [f"{channel}-{i}" for channel, i in itertools.product(channels, range(L))] + ["label"]


# %%
preprocessed_dataset = dataset.map(preprocess, batched=True, remove_columns=remove_columns)
print("done preprocessing!")
# %%
preprocessed_dataset.save_to_disk("eeg_mnist_preprocessed", num_proc=8)
