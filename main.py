import os
import torch
from torch.optim import AdamW  # ✅ পরিবর্তিত AdamW import এখানে
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader


# ডায়ালগ লোডার (DailyDialog থেকে)
def load_dialogues(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    dialogues = [line.strip().split(" __eou__ ")[:-1] for line in lines]
    return dialogues

# Dataset ক্লাস
class DialogDataset(Dataset):
    def __init__(self, dialogues, tokenizer, max_len=128):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = self.prepare_pairs()

    def prepare_pairs(self):
        pairs = []
        for dialog in self.dialogues:
            for i in range(len(dialog) - 1):
                input_text = dialog[i]
                target_text = dialog[i + 1]
                pairs.append((input_text, target_text))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_text, target_text = self.pairs[idx]

        input_enc = self.tokenizer(
            input_text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        target_enc = self.tokenizer(
            target_text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )

        input_ids = input_enc["input_ids"].squeeze()
        attention_mask = input_enc["attention_mask"].squeeze()
        labels = target_enc["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding tokens in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# ট্রেনিং ফাংশন
def train(model, tokenizer, dataloader, optimizer, device):
    model.train()
    loop = tqdm(dataloader, leave=True)
    total_loss = 0
    for batch in loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

# মডেল ইনফারেন্স (Generate response)
def chat(model, tokenizer, device, text, max_len=50):
    model.eval()
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=max_len,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.8,
    )
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ডায়ালগ লোড করো
    dialogues = load_dialogues("ijcnlp_dailydialog/dialogues_text.txt")

    # Pretrained tokenizer এবং মডেল (small model)
    model_name = "t5-small"  # CPU এর জন্য ছোট মডেল
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)

    dataset = DialogDataset(dialogues, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    epochs = 2  # কম্পিউটার ক্ষমতা অনুযায়ী বাড়াও

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        loss = train(model, tokenizer, dataloader, optimizer, device)
        print(f"Average loss: {loss:.4f}")

    # মডেল সেভ করো
    save_dir = "saved_model"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    # চ্যাট করা
    while True:
        text = input("\nYou: ")
        if text.lower() in ["exit", "quit"]:
            break
        response = chat(model, tokenizer, device, text)
        print(f"AI: {response}")

if __name__ == "__main__":
    main()
