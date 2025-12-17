# 🤖 Optimus Ask

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

Une application Full Stack de **Question-Réponse (QA)** utilisant un modèle Transformer (BERT) entraîné "From Scratch".
L'interface présente un petit robot animé qui réagit pendant l'inférence.

---

## 🚀 Démarrage Rapide

Le projet nécessite **deux terminaux** ouverts simultanément : un pour le Backend (API) et un pour le Frontend (React).

### 1️⃣ Terminal 1 : Le Backend (API Python)

Ce terminal charge le modèle PyTorch et lance le serveur FastAPI.

```bash
# 1. Aller dans le dossier backend
cd backend

# 2. Créer/Activer l'environnement virtuel (Recommandé)
# Windows :
python -m venv venv
.\venv\Scripts\activate
# Mac/Linux :
# python3 -m venv venv
# source venv/bin/activate

# 3. Installer les dépendances (si ce n'est pas déjà fait)
pip install torch transformers fastapi uvicorn

# 4. Lancer le serveur (avec uv si installé, sinon python directement)
uv run uvicorn main:app --reload
# OU
python -m uvicorn main:app --reload
```

> ✅ **Succès :** Le terminal affichera `Application startup complete`.
> L'API est accessible sur : `http://127.0.0.1:8000`

---

### 2️⃣ Terminal 2 : Le Frontend (React + Vite)

Ce terminal lance l'interface utilisateur.

```bash
# 1. Aller dans le dossier frontend
cd frontend

# 2. Installer les paquets (première fois seulement)
npm install

# 3. Lancer le site
npm run dev
```

> ✅ **Succès :** Le terminal affichera une URL locale (ex: `http://localhost:5173`).
> Ouvre ce lien dans ton navigateur pour utiliser le bot !

---

## 📂 Structure du Projet

```text
my-qa-app/
├── backend/
│   ├── ai/
│   │   ├── model.pth        # Poids du modèle entraîné
│   │   └── model_scratch.py # Architecture du Transformer
│   ├── main.py              # API FastAPI
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/      # Le Robot Kawaii
    │   └── App.jsx          # Logique principale
    └── package.json
```

## 🛠 Dépannage

* **Erreur `ModuleNotFoundError`** : Vérifie que ton environnement virtuel est bien activé (`(venv)` doit apparaître dans le terminal).
* **Erreur CORS / Network Error** : Vérifie que le Backend tourne bien. Si l'URL du backend a changé, mets à jour le `fetch` dans `App.jsx`.
* **Le robot ne change pas d'expression** : Assure-toi d'avoir renommé tes fichiers `.js` en `.jsx`.

---
