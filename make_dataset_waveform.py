import librosa
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
import torch
import datasets
from PIL import Image
from transformers import ASTConfig, ASTFeatureExtractor, ASTModel, CLIPModel, CLIPProcessor

def preprocess_waveform(inputs):
  S = 256
  C = len(channels)
  samples = np.zeros((S, C))
  for i, channel in enumerate(channels):
    samples[:, i] = np.array([inputs[f"{channel}-{j}"] for j in range(S)])
    for j in range(S):
      del inputs[f"{channel}-{j}"]
  inputs["pixel_values"] = samples
  return inputs



def main():
    dataset = datasets.load_dataset("DavidVivancos/MindBigData2022_MNIST_IN")
    channels = sorted(set(map(lambda x: x.split("-")[0], dataset["train"][0].keys())) - {'label'})
    dataset_train_preprocessed = dataset["train"].map(preprocess_waveform, num_proc=32)
    torch.tensor(dataset_train_preprocessed["pixel_values"][:10]).shape


if __name__ == "__main__":
    main()
