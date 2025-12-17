import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer
from ai.model_scratch import ExtractiveTransformer, MAX_LEN, DEVICE

# 1. Initialisation de l'app
app = FastAPI(title="Optimus Ask")

# Configuration CORS pour autoriser le frontend React
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"] permet à tout le monde de se connecter
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], # Autorise tous les verbes (GET, POST, etc.)
    allow_headers=["*"], # Autorise tous les headers
)

# 2. Chargement du Modèle et du Tokenizer (Au démarrage)
print("Loading Model & Tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Compatible avec vocab 30522

model = ExtractiveTransformer().to(DEVICE)

# Chargement des poids (Map location assure que ça marche même si tu n'as pas de GPU sur le serveur)
try:
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval() # Mode évaluation (désactive Dropout, etc.)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("WARNING: 'model.pth' not found. Make sure to place it in the backend folder.")

# 3. Définition du format de requête
class QAInput(BaseModel):
    context: str
    question: str

# 4. Logique d'Inférence
def predict_answer(context, question):
    # Préparation des inputs
    inputs = tokenizer.encode_plus(
        question,
        context,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask=attention_mask)

    # Récupération des meilleurs indices
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    # Si le modèle prédit la fin avant le début, c'est une erreur, on corrige
    if end_idx < start_idx:
        end_idx = start_idx

    # Décodage de la réponse
    answer_ids = input_ids[0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)
    
    return answer if answer.strip() else "Je n'ai pas trouvé de réponse..."

# 5. Endpoint API
@app.post("/predict")
async def get_answer(data: QAInput):
    try:
        response = predict_answer(data.context, data.question)
        # Simulation d'un petit délai pour voir l'animation (optionnel, à retirer en prod)
        # import time; time.sleep(2) 
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))