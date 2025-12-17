import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_scratch import ExtractiveTransformer, MAX_LEN

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 384
VOCAB_SIZE = 30522
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 1024
NUM_LAYERS = 6
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4

print(f"Device: {DEVICE}")

# --- FONCTION UTILITAIRE F1 ---
def calculate_token_f1(pred_start, pred_end, true_start, true_end):
    # Si le modèle prédit une fin avant le début -> F1 = 0
    if pred_end < pred_start:
        return 0.0
    
    # On crée des ensembles d'indices (ex: {10, 11, 12})
    pred_tokens = set(range(pred_start, pred_end + 1))
    true_tokens = set(range(true_start, true_end + 1))
    
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return 0.0
    
    # Nombre de tokens en commun
    common = len(pred_tokens.intersection(true_tokens))
    
    if common == 0:
        return 0.0
    
    precision = common / len(pred_tokens)
    recall = common / len(true_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

# --- 1. DATASETS ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data = load_dataset("rajpurkar/squad")
ds_train = data['train']
ds_val = data['validation']

class SquadExtractiveDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        try:
            answer_text = item['answers']['text'][0]
            answer_start = item['answers']['answer_start'][0]
        except IndexError:
            answer_text = ""
            answer_start = 0

        inputs = self.tokenizer(
            question,
            context,
            max_length=MAX_LEN,
            padding="max_length",
            truncation="only_second",
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        mask = inputs["attention_mask"].squeeze(0) 
        offsets = inputs["offset_mapping"].squeeze(0)
        
        answer_end = answer_start + len(answer_text)
        start_token_idx = 0
        end_token_idx = 0
        
        if answer_text:
            for i, (o_start, o_end) in enumerate(offsets):
                if o_start <= answer_start and o_end >= answer_start:
                    start_token_idx = i
                if o_start <= answer_end and o_end >= answer_end:
                    end_token_idx = i
                    break
            if end_token_idx < start_token_idx:
                end_token_idx = start_token_idx

        return input_ids, mask, torch.tensor(start_token_idx), torch.tensor(end_token_idx)

train_dataset = SquadExtractiveDataset(ds_train, tokenizer)
val_dataset = SquadExtractiveDataset(ds_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- 2. INITIALISATION ---
model = ExtractiveTransformer().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_val_loss = float('inf')
history_train_loss = []
history_val_loss = []
history_val_f1 = [] # Nouvelle liste pour stocker la F1

# --- 3. BOUCLE ---
print("Démarrage de l'entraînement...")

for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train()
    total_train_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
    
    for input_ids, mask, start_idx, end_idx in loop:
        input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
        start_idx, end_idx = start_idx.to(DEVICE), end_idx.to(DEVICE)
        
        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids, mask)
        
        loss = (criterion(start_logits, start_idx) + criterion(end_logits, end_idx)) / 2
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    avg_train_loss = total_train_loss / len(train_loader)
    history_train_loss.append(avg_train_loss)

    # --- VALIDATION (Loss + F1) ---
    model.eval()
    total_val_loss = 0
    total_f1 = 0
    
    with torch.no_grad():
        for input_ids, mask, start_idx, end_idx in val_loader:
            input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
            start_idx, end_idx = start_idx.to(DEVICE), end_idx.to(DEVICE)
            
            start_logits, end_logits = model(input_ids, mask)
            
            # Loss
            loss = (criterion(start_logits, start_idx) + criterion(end_logits, end_idx)) / 2
            total_val_loss += loss.item()
            
            # Calcul F1
            pred_start_batch = torch.argmax(start_logits, dim=1)
            pred_end_batch = torch.argmax(end_logits, dim=1)
            
            for i in range(input_ids.size(0)):
                f1 = calculate_token_f1(
                    pred_start_batch[i].item(), pred_end_batch[i].item(),
                    start_idx[i].item(), end_idx[i].item()
                )
                total_f1 += f1
            
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_f1 = total_f1 / len(val_dataset) # Moyenne sur tout le dataset
    
    history_val_loss.append(avg_val_loss)
    history_val_f1.append(avg_val_f1)

    print(f"Epoch {epoch+1} | Loss: {avg_val_loss:.4f} | F1 Score: {avg_val_f1:.4f}")

    # --- GRAPHIQUE (2 Subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graphique 1 : Loss (Doit descendre)
    ax1.plot(history_train_loss, label='Train Loss', color='blue')
    ax1.plot(history_val_loss, label='Val Loss', color='red')
    ax1.set_title('Loss (Erreur)')
    ax1.grid(True)
    ax1.legend()
    
    # Graphique 2 : F1 Score (Doit monter vers 1.0)
    ax2.plot(history_val_f1, label='Val F1 Score', color='green')
    ax2.set_title('F1 Score (Précision)')
    ax2.grid(True)
    ax2.legend()
    
    plt.savefig('monitoring_metrics.png')
    plt.close()

    # --- SAUVEGARDE ---
    torch.save(model.state_dict(), "squad_scratch_extractive_last.pth")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "squad_scratch_extractive_best.pth")
        print(f"Nouveau record Loss : {best_val_loss:.4f}")

print("Entraînement terminé.")