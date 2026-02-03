from __future__ import annotations

import argparse
from pathlib import Path

import tiktoken
from torch.utils.data import DataLoader

from dataset import GPTDatasetV1


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
):
    # 1) Tokenizer initialisieren (wie im Buch: GPT-2 Encoding)
    tokenizer = tiktoken.get_encoding("gpt2")

    # 2) Dataset erzeugen (macht aus Text -> (input,target)-Paare)
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 3) DataLoader erzeugen (stapelt Beispiele zu Batches)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def main():
    parser = argparse.ArgumentParser(description="GPT Dataset/DataLoader Demo (aus Buch-Listings)")
    parser.add_argument("--file", type=str, default="the-verdict.txt", help="Pfad zur Textdatei")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch-Größe")
    parser.add_argument("--max-length", type=int, default=4, help="Kontextlänge (Chunk-Länge)")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-Window-Schrittweite")
    parser.add_argument("--shuffle", action="store_true", help="Batches mischen")
    parser.add_argument("--no-drop-last", action="store_true", help="Letzten unvollständigen Batch behalten")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader Worker (Windows meist 0)")
    parser.add_argument("--show-batches", type=int, default=1, help="Wie viele Batches ausgeben")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        # Fallback: Damit es IMMER läuft, selbst ohne Datei
        raw_text = "Hello world! This is a tiny demo text for GPTDatasetV1.\n"
        print(f"[Info] Datei '{path}' nicht gefunden – nutze Demo-Text.")
    else:
        raw_text = path.read_text(encoding="utf-8")

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=args.batch_size,
        max_length=args.max_length,
        stride=args.stride,
        shuffle=args.shuffle,
        drop_last=not args.no_drop_last,
        num_workers=args.num_workers,
    )

    print(f"[Info] Anzahl Beispiele im Dataset: {len(dataloader.dataset)}")
    print(f"[Info] Batch Size: {args.batch_size}, max_length: {args.max_length}, stride: {args.stride}")

    data_iter = iter(dataloader)

    for n in range(args.show_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            print("[Info] Keine weiteren Batches.")
            break

        input_ids, target_ids = batch
        print(f"\nBatch {n+1}:")
        print("input_ids :", input_ids)
        print("target_ids:", target_ids)


if __name__ == "__main__":
    main()
