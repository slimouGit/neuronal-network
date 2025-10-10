# train_and_save.py
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model import NeuralNetwork  # dein Modell aus model.py importieren

# Zufällige Reproduzierbarkeit
torch.manual_seed(42)

# ------------------------------------------------------------
# 1) Device automatisch wählen (GPU oder CPU)
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Trainingsgerät: {device}")

# ------------------------------------------------------------
# 2) Trainingsdaten (verzehnfacht)
# ------------------------------------------------------------
X_train = torch.tensor([
    [ 1.0,  1.0],
    [ 2.0,  2.0],
    [ 3.0,  3.0],
    [-1.0, -1.0],
    [-2.0, -2.0],
] * 10, dtype=torch.float32)  # 10x Wiederholung = 50 Samples

y_train = torch.tensor([0, 0, 0, 1, 1] * 10, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=5, shuffle=True)

# ------------------------------------------------------------
# 3) Modell und Optimizer
# ------------------------------------------------------------
model = NeuralNetwork(num_inputs=2, num_outputs=2).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# ------------------------------------------------------------
# 4) Trainingsschleife
# ------------------------------------------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (features, labels) in enumerate(train_loader, start=1):
        features, labels = features.to(device), labels.to(device)

        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

# ------------------------------------------------------------
# 5) Modell speichern
# ------------------------------------------------------------
torch.save(model.state_dict(), "model.pth")
print("\n💾 Modell gespeichert als 'model.pth'")

# ------------------------------------------------------------
# 6) Modell laden und auf Gerät übertragen
# ------------------------------------------------------------
loaded_model = NeuralNetwork(num_inputs=2, num_outputs=2)
loaded_model.load_state_dict(torch.load("model.pth", map_location=device))
loaded_model = loaded_model.to(device)
loaded_model.eval()
print("📂 Modell erfolgreich geladen und auf Gerät übertragen.")

# ------------------------------------------------------------
# 7) Evaluation (inference)
# ------------------------------------------------------------
with torch.no_grad():
    X_test = torch.tensor([
        [ 1.5,  1.5],  # zweibeinig
        [-1.5, -1.5],  # vierbeinig
        [ 2.5,  2.5],
        [-2.5, -2.5],
    ], dtype=torch.float32).to(device)

    outputs = loaded_model(X_test)
    predictions = torch.argmax(outputs, dim=1)

print("\nVorhersagen:")
for x, pred in zip(X_test.cpu(), predictions.cpu()):
    label = "zweibeinig" if pred.item() == 0 else "vierbeinig"
    print(f"Input: {x.tolist()} → {label}")

print("\n✅ Testdurchlauf abgeschlossen.")
