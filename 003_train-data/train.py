# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from model import NeuralNetwork  # <-- Importiert dein Modell

# -----------------------------
# 1. Trainingsdaten (kleines Demo-Set)
# -----------------------------
X_train = torch.tensor(
    [[ 1.0,  1.0],
     [ 2.0,  2.0],
     [ 3.0,  3.0],
     [-1.0, -1.0],
     [-2.0, -2.0]], dtype=torch.float32
)
y_train = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=2, shuffle=True)

# -----------------------------
# 2. Modell und Optimizer
# -----------------------------
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# -----------------------------
# 3. Trainingsschleife
# -----------------------------
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Batch {batch_idx+1}/{len(train_loader)} | "
            f"Loss: {loss.item():.4f}"
        )

# -----------------------------
# 4. Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    outputs = model(X_train)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = torch.sum(predictions == y_train).item() / len(y_train)

print("\nVorhersagen:", predictions.tolist())
print("Echte Labels:", y_train.tolist())
print(f"Accuracy: {accuracy:.2f}")

# -----------------------------
# 5. Optional: allgemeine Accuracy-Funktion
# -----------------------------
def compute_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            logits = model(features)
            preds = torch.argmax(logits, dim=1)
            correct += torch.sum(preds == labels).item()
            total += len(labels)
    return correct / total

print("Accuracy (compute_accuracy):", compute_accuracy(model, train_loader))
