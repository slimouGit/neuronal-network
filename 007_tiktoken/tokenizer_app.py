from importlib.metadata import version

import tiktoken




def show_header():
    print("=" * 60)
    print("  GPT-2 Tokenizer Playground (tiktoken)")
    print("=" * 60)
    try:
        print(f"tiktoken version: {version('tiktoken')}")
    except Exception:
        print("tiktoken version: (konnte nicht ermittelt werden)")
    print()


def get_tokenizer():
    # wie im Buch: GPT-2-Tokenisierung
    return tiktoken.get_encoding("gpt2")


def encode_text(tokenizer):
    print("\n--- Text → Token-IDs ---")
    print("Gib deinen Text ein. Eine leere Zeile beendet die Eingabe.")
    print("(Du kannst z.B. das Beispiel aus dem Buch nehmen.)")
    print()

    # Mehrzeilige Eingabe
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    text = "\n".join(lines)

    if not text.strip():
        print("Kein Text eingegeben.\n")
        return

    # Beispiel aus dem Buch erlaubt <|endoftext|> als spezielles Token
    integers = tokenizer.encode(
        text,
        allowed_special={"<|endoftext|>"}
    )

    print("\nToken-IDs:")
    print(integers)
    print(f"\nAnzahl Tokens: {len(integers)}\n")


def decode_tokens(tokenizer):
    print("\n--- Token-IDs → Text ---")
    print("Gib Token-IDs als durch Leerzeichen getrennte Liste ein.")
    print("Beispiel: 15496 11 466 345 588 8887 30 220 ...")
    print()

    raw = input("Token-IDs: ").strip()
    if not raw:
        print("Keine Token-IDs eingegeben.\n")
        return

    try:
        integers = [int(x) for x in raw.split()]
    except ValueError:
        print("Fehler: Bitte nur ganze Zahlen eingeben.\n")
        return

    text = tokenizer.decode(integers)
    print("\nDekodierter Text:")
    print(text)
    print()


def main():
    show_header()
    tokenizer = get_tokenizer()

    # Beispieltext aus dem Buch – wird einmal automatisch gezeigt
    example_text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
        "of someunknownPlace."
    )
    example_ids = tokenizer.encode(
        example_text,
        allowed_special={"<|endoftext|>"}
    )
    print("Beispiel aus dem Buch:")
    print("Text:")
    print(example_text)
    print("\nToken-IDs:")
    print(example_ids)
    print("\nDekodiert:")
    print(tokenizer.decode(example_ids))
    print("\n" + "=" * 60)

    while True:
        print("Menü:")
        print("  [1] Text in Token-IDs umwandeln")
        print("  [2] Token-IDs in Text zurückwandeln")
        print("  [3] Beenden")
        choice = input("Deine Auswahl: ").strip()

        if choice == "1":
            encode_text(tokenizer)
        elif choice == "2":
            decode_tokens(tokenizer)
        elif choice == "3":
            print("Tschüss! 👋")
            break
        else:
            print("Bitte 1, 2 oder 3 eingeben.\n")


if __name__ == "__main__":
    main()
