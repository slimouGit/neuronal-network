# pytorch_training_loop_minimal.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(123)

# --- 1) Dummy-Daten: 2 Features, 2 Klassen (0/1) ----------------------------
X_train = torch.tensor(
    [[ 1.0,  1.0],
     [ 2.0,  2.0],
     [ 3.0,  3.0],
     [-1.0, -1.0],
     [-2.0, -2.0]], dtype=torch.float32
)
y_train = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=2, shuffle=True)

# --- 2) Einfaches Modell ------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs=2, num_outputs=2):
        super().__init__()
        # Eine einzige lineare Schicht: Logit-Ausgabe für 2 Klassen
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        return self.linear(x)  # Logits; CrossEntropyLoss erwartet rohe Logits

model = NeuralNetwork(num_inputs=2, num_outputs=2)

# --- 3) Optimizer -------------------------------------------------------------
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# --- 4) Training --------------------------------------------------------------
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)                     # Vorwärtslauf
        loss = F.cross_entropy(logits, labels)       # Klassifikationsverlust

        optimizer.zero_grad()                        # (1) Gradienten nullen
        loss.backward()                              # (2) Backprop
        optimizer.step()                             # (3) Update

        # Logging wie im Buch
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d} | "
            f"Batch {batch_idx:03d}/{len(train_loader):03d} | "
            f"Train Loss: {loss.item():.2f}"
        )

# --- 5) Evaluation: Logits, Softmax, Argmax ----------------------------------
model.eval()
with torch.no_grad():
    outputs = model(X_train)                  # Logits
    print("\nLogits:")
    print(outputs)

    probas = torch.softmax(outputs, dim=1)    # Wahrscheinlichkeiten
    print("\nWahrscheinlichkeiten (Softmax):")
    print(probas)

    predictions = torch.argmax(probas, dim=1) # oder direkt argmax(outputs, dim=1)
    print("\nVorhersagen (Argmax):")
    print(predictions)

    compare = (predictions == y_train)
    print("\nTreffervergleich pro Sample:")
    print(compare)
    print("\nAnzahl korrekter Vorhersagen:", torch.sum(compare).item())

# --- 6) Allgemeine Accuracy-Funktion (wie im Buch) ----------------------------
def compute_accuracy(model: nn.Module, dataloader: DataLoader) -> float:
    model = model.eval()
    correct = 0
    total_examples = 0
    with torch.no_grad():
        for _, (features, labels) in enumerate(dataloader):
            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            compare = (labels == preds)                # (1) True/False
            correct += torch.sum(compare).item()       # (2) True zählen
            total_examples += len(compare)
    return correct / total_examples                    # (3) Anteil korrekter

print("\nAccuracy (Train):", compute_accuracy(model, train_loader))
