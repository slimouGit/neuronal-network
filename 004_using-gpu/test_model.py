import torch
from model import NeuralNetwork

class ModelTester:
    """
    Lädt ein gespeichertes PyTorch-Modell und testet es auf neuen Daten.
    Unterstützt automatisch CPU oder GPU.
    """

    def __init__(self, model_path: str, num_inputs: int, num_outputs: int):
        # Erkenne Gerät
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📦 Lade Modell auf Gerät: {self.device}")

        # Modell initialisieren und laden
        self.model = NeuralNetwork(num_inputs, num_outputs)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"✅ Modell erfolgreich geladen: '{model_path}'\n")

    def predict(self, X: torch.Tensor):
        """
        Gibt Klassenvorhersagen für die Eingabedaten zurück.
        """
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu()

    def predict_proba(self, X: torch.Tensor):
        """
        Gibt Softmax-Wahrscheinlichkeiten (pro Klasse) zurück.
        """
        X = X.to(self.device)
        with torch.no_grad():
            logits = self.model(X)
            probas = torch.softmax(logits, dim=1)
        return probas.cpu()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor):
        """
        Berechnet die Accuracy des Modells.
        """
        preds = self.predict(X)
        acc = torch.sum(preds == y).item() / len(y)
        return acc


# ---------------------------------------------------------
# Beispielhafte Nutzung
# ---------------------------------------------------------
if __name__ == "__main__":
    # Initialisiere Tester
    tester = ModelTester(model_path="model.pth", num_inputs=2, num_outputs=2)

    # Beispiel-Testdaten
    X_test = torch.tensor([
        [ 1.5,  1.5],  # zweibeinig erwartet (0)
        [ 2.5,  2.5],  # zweibeinig (0)
        [-1.5, -1.5],  # vierbeinig (1)
        [-2.5, -2.5],  # vierbeinig (1)
    ], dtype=torch.float32)

    y_test = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    # Vorhersagen
    preds = tester.predict(X_test)
    probas = tester.predict_proba(X_test)
    acc = tester.evaluate(X_test, y_test)

    # Ausgabe
    label_names = {0: "zweibeinig", 1: "vierbeinig"}
    print("\n🔍 Vorhersagen:")
    for x, p, pr in zip(X_test, preds, probas):
        print(f"Eingabe: {x.tolist()} → {label_names[int(p)]} "
              f"(P={pr.tolist()})")

    print(f"\n🎯 Genauigkeit auf Testdaten: {acc:.2f}")
