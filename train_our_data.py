
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

# %%
# B, C, H, W
C = 4
H, W = 168, 400
S = max(H, W)


# %%
tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-large-patch14")

# %%


hidden_size = 16
configuration = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")
configuration.update(
  {
    "image_size": S,
    "num_channels": C,
    "num_hidden_layers": 4,
    "hidden_size": hidden_size,
    "intermediate_size": 4 * hidden_size,
    "num_attention_heads": 8,
    "projection_dim": 768,
  }
)
vision_model = CLIPVisionModelWithProjection(configuration)


# %%
model = TrainableCLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


# %%
model.vision_model = vision_model.vision_model
model.visual_projection = nn.Linear(hidden_size, model.projection_dim, bias=False)
model.config.vision_config = configuration
# %%
for p in model.text_model.parameters():
  p.requires_grad = False
for p in model.text_projection.parameters():
  p.requires_grad = False


# %%
print(model)
print(f"All Parameters: {sum(p.numel() for p in model.parameters()) / 1e6} M")
print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e3} K")


# %%
preprocessed_dataset = load_dataset("our_data", keep_in_memory=True)

# %%
preprocessed_dataset['train'], preprocessed_dataset['validation'] = preprocessed_dataset['train'].train_test_split(.1).values()


# %%
training_args = TrainingArguments(
    output_dir="clep_trainer_our_data",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=1,
    tf32=True,
    num_train_epochs=10,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    torch_compile=True,
    label_names=["labels"],
)

import evaluate
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    outputs, labels = eval_pred
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    prediction = probs.argmax(dim=1)
    accuracy = accuracy_metric.compute(predictions=prediction, references=labels)
    return {"accuracy": accuracy}


# %%
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
