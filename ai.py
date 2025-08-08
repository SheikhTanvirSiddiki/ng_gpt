from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch

# Load dataset
dataset = load_dataset("daily_dialog")

# Load tokenizer and model
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(example):
    # Combine dialog history into a single prompt
    inputs = "dialog: " + example["dialog"][0]  # Only first line of dialog
    targets = example["dialog"][-1] if len(example["dialog"]) > 1 else "..."

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=64, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# Preprocess dataset
tokenized_dataset = dataset["train"].map(preprocess_function, batched=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./dialogue_model",
    evaluation_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    save_strategy="epoch"
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./dialogue_model")
tokenizer.save_pretrained("./dialogue_model")

print("✅ Model trained and saved successfully!")
