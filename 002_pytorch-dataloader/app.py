# %%
import torch
from torch.utils.data import Dataset, DataLoader

# ----- 1) Kleinen Übungsdatensatz erstellen -----
X_train = torch.tensor([
    [-1.2,  3.1],
    [-0.9,  2.9],
    [-0.5,  2.6],
    [ 2.3, -1.1],
    [ 2.7, -1.5],
], dtype=torch.float32)

# Labels als LongTensor, weil viele Loss-Funktionen (z. B. CrossEntropyLoss) int-Klassen erwarten
y_train = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

X_test = torch.tensor([
    [-0.8,  2.8],
    [ 2.6, -1.6],
], dtype=torch.float32)

y_test = torch.tensor([0, 1], dtype=torch.long)

# ----- 2) Benutzerdefinierte Dataset-Klasse -----
class ToyDataset(Dataset):
    def __init__(self, X, y):
        """
        X: Tensor der Features mit Form [N, D]
        y: Tensor der Labels mit Form [N]
        """
        assert len(X) == len(y), "Features und Labels müssen gleich lang sein."
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        """
        Gibt ein Tupel (x_i, y_i) zurück.
        """
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        """
        Anzahl der Beispiele im Dataset.
        """
        return self.labels.shape[0]

# Instanzen für Train/Test
train_ds = ToyDataset(X_train, y_train)
test_ds  = ToyDataset(X_test,  y_test)

print("Len train_ds:", len(train_ds))  # sollte 5 sein
print("Len test_ds :", len(test_ds))   # sollte 2 sein

# ----- 3) DataLoader einrichten -----
torch.manual_seed(123)  # für reproduzierbares Shuffling

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,    # Größe der Mini-Batches
    shuffle=True,    # mischen pro Epoche
    num_workers=0,   # 0 = Laden im Hauptprozess (für kleine Demos ok)
)

# Für Testdaten mischt man i. d. R. nicht
test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,
    num_workers=0,
)

# Variante: letztes unvollständiges Batch verwerfen (wie im Buch gezeigt)
train_loader_drop_last = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    drop_last=True,  # bei 5 Samples bleiben hier nur 2 Batches à 2 erhalten
)

# ----- 4) Über die Loader iterieren und Batches anzeigen -----
print("\nBatches aus train_loader:")
for idx, (x, y) in enumerate(train_loader, start=1):
    print(f"Batch {idx}: X.shape={x.shape}, y={y.tolist()}\n{x}\n")

print("Batches aus train_loader_drop_last (letzten unvollständigen Batch verwerfen):")
for idx, (x, y) in enumerate(train_loader_drop_last, start=1):
    print(f"Batch {idx}: X.shape={x.shape}, y={y.tolist()}\n{x}\n")

print("Batches aus test_loader:")
for idx, (x, y) in enumerate(test_loader, start=1):
    print(f"Batch {idx}: X.shape={x.shape}, y={y.tolist()}\n{x}\n")
