import torch

# Ein mehrschichtiges Perzeptron mit zwei versteckten Schichten
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


# Beispiel: Modell anlegen (50 Eingaben, 3 Ausgaben)
model = NeuralNetwork(50, 3)
print(model)

# Beispiel-Eingabe (Batchgröße 1, 50 Features)
X = torch.randn(1, 50)

# Inferenz ohne Gradienten: Logits
with torch.no_grad():
    out_logits = model(X)
    print(out_logits)

# Optional: in Wahrscheinlichkeiten umrechnen
with torch.no_grad():
    out_probs = torch.softmax(model(X), dim=1)
    print(out_probs)
