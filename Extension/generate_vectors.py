import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import random
from llama_wrapper import LlamaWrapper
from training_args import parse_args
from typing import List
from utils.tokenize import tokenize_llama_base

def get_vector_dir(behavior):
    """Returns the directory path where vectors for the behavior are stored."""
    return f"./vectors/{behavior}"

def get_vector_path(behavior, layer, model_name):
    """Generates a save path for the vector based on behavior, layer, and model name."""
    sanitized_model_name = model_name.replace("/", "_")  # Ensure path-safe names
    vector_dir = get_vector_dir(behavior)
    os.makedirs(vector_dir, exist_ok=True)  # Ensure directory exists
    return os.path.join(vector_dir, f"{sanitized_model_name}_layer{layer}_trainable.pt")

def pad_sequences(sequences, padding_value=0):
    max_len = max(seq.size(1) for seq in sequences)
    padded_sequences = []
    attention_masks = []
    for seq in sequences:
        pad_length = max_len - seq.size(1)
        padded_seq = F.pad(seq, (0, pad_length), value=padding_value)
        attention_mask = torch.ones_like(padded_seq, dtype=torch.bool)
        attention_mask[:, -pad_length:] = 0
        padded_sequences.append(padded_seq)
        attention_masks.append(attention_mask)

    return torch.cat(padded_sequences), torch.cat(attention_masks)

class ComparisonDataset(Dataset):
    def __init__(self, data_path, model_name_path, cache_dir):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def prompt_to_tokens(self, instruction, model_output):
        tokens = tokenize_llama_base(
            self.tokenizer,
            user_input=instruction,
            model_output=model_output,
        )
        return torch.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens

def custom_collate(batch):
    p_tokens, n_tokens = zip(*batch)
    p_padded, p_mask = pad_sequences(p_tokens)
    n_padded, n_mask = pad_sequences(n_tokens)
    return p_padded, n_padded, p_mask, n_mask

def train_steering_vector(
    layers: List[int],
    behavior: str,
    model: LlamaWrapper,
    data_path: str,
    cache_dir: str,
    learning_rate: float = 1e-3,
    num_epochs: int = 15,
    batch_size: int = 8,
    beta: float = 0.1,
    lr_step_size: int = 5,
    lr_gamma: float = 0.5
):
    # Create dataset and dataloader
    dataset = ComparisonDataset(
        data_path,
        cache_dir=cache_dir,
        model_name_path=model.model_name,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    loss_arr = []

    for layer in layers:
        steering_vector = nn.Parameter(torch.zeros(model.model.config.hidden_size, device=model.device))
        optimizer = optim.AdamW([steering_vector], lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        multiplier = 1

        for epoch in range(num_epochs):
            total_loss = 0
            scaler = torch.amp.GradScaler('cuda')
            for p_tokens, n_tokens, _, _ in tqdm(dataloader, desc=f"Layer {layer}, Epoch {epoch+1}"):
                with torch.amp.autocast('cuda'):
                    p_tokens = p_tokens.to(model.device)
                    n_tokens = n_tokens.to(model.device)

                    model.reset_all()
                    base_logits_pos = model.get_logits(p_tokens)
                    base_logits_neg = model.get_logits(n_tokens)
                    base_log_prob_pos = torch.log_softmax(base_logits_pos, dim=-1)
                    base_log_prob_neg = torch.log_softmax(base_logits_neg, dim=-1)

                    model.reset_all()
                    model.set_add_activations(layer, steering_vector)
                    steered_logits_pos = model.get_logits(p_tokens)
                    steered_logits_neg = model.get_logits(n_tokens)
                    steered_log_prob_pos = torch.log_softmax(steered_logits_pos, dim=-1)
                    steered_log_prob_neg = torch.log_softmax(steered_logits_neg, dim=-1)

                    rel_log_neg = steered_log_prob_neg - base_log_prob_neg
                    rel_log_pos = steered_log_prob_pos - base_log_prob_pos
                    logits = rel_log_pos - rel_log_neg

                    logits = multiplier * beta * logits

                    loss = -torch.mean(F.logsigmoid(logits))
                    total_loss += loss.item()

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()
            print(f"Layer {layer}, Epoch {epoch+1}, Average Loss: {total_loss / len(dataloader)}")
            loss_arr.append(total_loss / len(dataloader))
            print(f"Current LR: {scheduler.get_last_lr()}")

        save_path = get_vector_path(behavior, layer, model.model_name)
        torch.save(steering_vector.detach().cpu(), save_path)
        print(f"Saved steering vector for layer {layer} to {save_path}")

    print(loss_arr)

def generate_trainable_steering_vectors(args):
    model = LlamaWrapper(cache_dir=args.cache_dir, model_name=args.model_name)

    train_steering_vector(
        layers=args.layers,
        behavior=args.behavior,
        model=model,
        data_path=args.data_path,
        cache_dir=args.cache_dir,
    )

if __name__ == "__main__":
    args = parse_args()
    generate_trainable_steering_vectors(args)
