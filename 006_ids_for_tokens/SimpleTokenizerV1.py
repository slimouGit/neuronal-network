# We'll implement a simple tokenizer inspired by the book photos and run a quick demo.
# Then we'll also save it as a standalone script the user can download and run locally.

import re
from pathlib import Path

class SimpleTokenizerV1:
    """
    A very small, whitespace+punctuation tokenizer with reversible decoding.

    - Splits on punctuation and whitespace but *keeps* the punctuation as tokens.
    - Filters out empty tokens and pure-space fragments.
    - Encodes by mapping tokens -> integers using a given vocabulary.
    - Decodes by mapping integers -> tokens and then "detaching" spaces
      that appear before punctuation.
    """
    def __init__(self, vocab: dict[str, int]):
        # store both directions for fast lookup
        self.str_to_int = dict(vocab)
        self.int_to_str = {i: s for s, i in vocab.items()}

        # same regex used by encode() to split
        # capture group ensures punctuation is kept as separate tokens
        self._split_re = re.compile(r'([.,!?\"“”()\'—;:]+|\s)')

    def encode(self, text: str) -> list[int]:
        # split into tokens (punctuation and whitespace kept)
        parts = self._split_re.split(text)
        # drop empty strings and pure-space tokens
        tokens = [p for p in parts if p and not p.isspace()]

        # look up ids; raise KeyError if unknown to make issues obvious
        try:
            return [self.str_to_int[t] for t in tokens]
        except KeyError as e:
            missing = str(e).strip("'")
            # provide a helpful error message
            raise KeyError(f"Token not in vocabulary: {missing!r}. "
                           f"Build your vocab with this token or add an <unk> id.") from None

    def decode(self, ids: list[int]) -> str:
        # map ids back to tokens and join with spaces
        tokens = [self.int_to_str[i] for i in ids]
        text = " ".join(tokens)
        # remove spaces that incorrectly appear before punctuation
        text = re.sub(r'\s+([.,!?\"””)\'];:])', r'\1', text)
        # fix spaces after opening quotes/brackets
        text = re.sub(r'([(\"“])\s+', r'\1', text)
        return text

def build_vocab_from_text(text: str) -> dict[str, int]:
    """
    Build a simple token->id vocabulary using the same splitting rule as the tokenizer.
    """
    split_re = re.compile(r'([.,!?\"“”()\'—;:]+|\s)')
    parts = split_re.split(text)
    tokens = [p for p in parts if p and not p.isspace()]
    # stable order of first appearance
    seen = {}
    for tok in tokens:
        if tok not in seen:
            seen[tok] = len(seen)
    return seen

# --- Demo ----------------------------------------------------------

training_text = (
    # Use the same example sentence from the screenshots to construct a vocab
    "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride."
)

failed_text = (
"Hello, my name is Bunny. I like to hop!"
)

# Build the vocabulary from the training text
vocab = build_vocab_from_text(training_text)
tokenizer = SimpleTokenizerV1(vocab)

# Encode the same text (so all tokens are known)
ids = tokenizer.encode(training_text)
decoded = tokenizer.decode(ids)

# failed_ids = tokenizer.encode(failed_text)
# failed_decoded = tokenizer.decode(failed_ids)

print("Vocabulary (first 20 items):")
print(list(vocab.items())[:20])
print("\nEncoded IDs:")
print(ids)
print("\nDecoded back:")
print(decoded)
# print(failed_decoded)


