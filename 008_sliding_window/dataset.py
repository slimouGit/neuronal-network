from __future__ import annotations

import torch
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    """
    Erzeugt (input_ids, target_ids) Paare aus einem langen Token-Stream.

    - input_ids:  [t_i, t_{i+1}, ..., t_{i+max_length-1}]
    - target_ids: [t_{i+1}, ..., t_{i+max_length}]  (um 1 verschoben)
    """

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int) -> None:
        self.input_ids = []
        self.target_ids = []

        # 1) Gesamten Text in Token-IDs umwandeln
        token_ids = tokenizer.encode(txt)

        # 2) Sliding Window: in überlappende Chunks schneiden
        #    Achtung: Wir brauchen target_chunk bis i+max_length (inkl. shift),
        #    deshalb iterieren wir bis len(token_ids) - max_length - 1 (implizit über slicing)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        # Anzahl der Trainingsbeispiele (Chunks)
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        # Ein einzelnes Trainingsbeispiel zurückgeben
        return self.input_ids[idx], self.target_ids[idx]
