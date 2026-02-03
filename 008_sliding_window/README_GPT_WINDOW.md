# GPT-WINDOW

**Sliding-Window-Datasets & DataLoader für GPT-Modelle (PyTorch)**

GPT-WINDOW ist eine kleine, bewusst transparente Lern- und Experimentier-App. Sie zeigt Schritt für Schritt, wie aus reinem Text Input/Target-Paare für Next-Token-Prediction entstehen – genau so, wie es klassische GPT-Modelle erwarten.

Die App orientiert sich eng an typischen Buch-Listings (Dataset + DataLoader) und ist ideal, um:

- GPT-Training wirklich zu verstehen
- eigene Token-Datasets zu bauen
- Sliding-Window-Logik zu experimentieren (Kontextgröße, Stride, Überlappung)

---

## 1. Grundidee

Ein GPT lernt immer dasselbe:

Vorhersage des nächsten Tokens basierend auf vorherigen Tokens.

Input:  t0, t1, t2, …, tn  
Target: t1, t2, t3, …, tn+1

GPT-WINDOW automatisiert genau diesen Schritt – für beliebig langen Text.

---

## 2. Projektstruktur

gpt-window/
- app.py
- dataset.py
- requirements.txt
- the-verdict.txt
- README.md

---

## 3. Installation

python -m venv .venv

Windows:
.\.venv\Scripts\Activate.ps1

Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

---

## 4. Zentrale Komponenten

Tokenisierung:
tiktoken.get_encoding("gpt2")

Dataset: GPTDatasetV1
Erzeugt Sliding-Window-Paare:

Input:  t0 t1 t2 t3  
Target: t1 t2 t3 t4

---

## 5. App starten

python app.py --file the-verdict.txt --batch-size 1 --max-length 4 --stride 1

---

## 6. Beispielausgabe

input_ids : tensor([[  40,  367, 2885, 1464]])
target_ids: tensor([[ 367, 2885, 1464, 1807]])

---

## 7. Erweiterungen

- Trainingsloop
- Mini-Transformer
- GPU-Support
- Attention Masks

---

## 8. Leitsatz

LLMs sind keine Blackbox – sie sind saubere Datenpipelines mit Matrix-Multiplikation.
