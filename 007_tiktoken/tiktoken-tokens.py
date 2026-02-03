import tiktoken
import json

enc = tiktoken.get_encoding("o200k_base")
vocab_size = enc.n_vocab
print("Vokabulargröße:", vocab_size)

tokens = []
for token_id in range(vocab_size):
    try:
        token_bytes = enc.decode_bytes([token_id])
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = repr(token_bytes)
        tokens.append((token_id, token_str))
    except KeyError:
        # Skip invalid token IDs
        continue

for tid, t in tokens[:1000]:
    print(tid, t)

with open("tiktoken_vocab.json", "w", encoding="utf-8") as f:
    json.dump(tokens, f, ensure_ascii=False, indent=2)