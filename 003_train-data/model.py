# model.py
import torch

class NeuralNetwork(torch.nn.Module):
    """
    Einfaches Feedforward-Netz mit zwei versteckten Schichten.
    - num_inputs:  Anzahl der Eingabefeatures
    - num_outputs: Anzahl der Ausgabeklassen
    """

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1. versteckte Schicht
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2. versteckte Schicht
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # Ausgabeschicht
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        """
        Forward-Pass: Eingabe -> Netzwerk -> Logits
        """
        logits = self.layers(x)
        return logits


# Beispielhafte Initialisierung (kann entfallen, wenn das Modell importiert wird)
if __name__ == "__main__":
    model = NeuralNetwork(50, 3)
    print(model)
