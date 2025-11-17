class SimpleTokenizerV2:
    """
    Tokenizer that uses text from a DummyTextLoader instance or a plain string.
    Builds vocabulary from the provided text.
    """
    def __init__(self, source):
        import re
        if hasattr(source, "raw_text"):
            self.text = source.raw_text
        elif isinstance(source, str):
            self.text = source
        else:
            raise ValueError("Source must be a DummyTextLoader or a string.")
        self._split_re = re.compile(r'([.,!?\"“”()\'—;:]+|\s)')
        self.vocab = self._build_vocab_from_text(self.text)
        self.str_to_int = dict(self.vocab)
        self.int_to_str = {i: s for s, i in self.vocab.items()}

    def _build_vocab_from_text(self, text):
        parts = self._split_re.split(text)
        tokens = [p for p in parts if p and not p.isspace()]
        seen = {}
        for tok in tokens:
            if tok not in seen:
                seen[tok] = len(seen)
        return seen

    def encode(self, text: str) -> list[int]:
        parts = self._split_re.split(text)
        tokens = [p for p in parts if p and not p.isspace()]
        try:
            return [self.str_to_int[t] for t in tokens]
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"Token not in vocabulary: {missing!r}.") from None

    def decode(self, ids: list[int]) -> str:
        import re
        tokens = [self.int_to_str[i] for i in ids]
        text = " ".join(tokens)
        text = re.sub(r'\s+([.,!?\"””)\'];:])', r'\1', text)
        text = re.sub(r'([(\"“])\s+', r'\1', text)
        return text