import logging

from datasets import load_dataset
from transformers import (
    CLIPProcessor,
    CLIPTokenizerFast,
    CLIPVisionConfig,
    CLIPVisionModelWithProjection,
    Trainer,
    TrainingArguments,
)
from torch import nn
from trainable_clip_model import TrainableCLIPModel
logging.basicConfig(filename='train.log', stream=sys.stdout, level=logging.INFO)

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
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

# %%
hidden_size = 64
configuration = CLIPVisionConfig(
    image_size=max(H, W),
    num_channels=C,
    num_hidden_layers=1,
    # hidden_size (`int`, *optional*, defaults to 768):
    #     Dimensionality of the encoder layers and the pooler layer.
    hidden_size=hidden_size,
    # intermediate_size (`int`, *optional*, defaults to 3072):
    #     Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
    intermediate_size=4*hidden_size,
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
model.visual_projection = nn.Linear(hidden_size, model.projection_dim, bias=False)

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
preprocessed_dataset = load_dataset("eeg_mnist_preprocessed", keep_in_memory=True)

# %%
preprocessed_dataset['train'], preprocessed_dataset['validation'] = preprocessed_dataset['train'].train_test_split(.1).values()


# %%
training_args = TrainingArguments(
    output_dir="clep_trainer",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    logging_steps=1,
    tf32=True,
    num_train_epochs=5,
    learning_rate=1e-3,
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
