from importlib.metadata import version

import tiktoken

print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

# text = (
#     "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
#     "of someunknownPlace."
# )
text = (
    "Eintracht vom Main <|endoftext|> Nur du sollst heute siegen "
    "Eintracht vom Main <|endoftext|> Weil wir dich so lieben"
)

# Allow the special token
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

print("altes decode: " + tokenizer.decode([617, 34680, 27271]))
print("altes decode: " + tokenizer.decode([4252, 18250]))
print("neues decode: " + tokenizer.decode([36, 600, 81, 19725, 20918, 8774]))
print("random decode: " + tokenizer.decode([1234, 5678, 5678]))

print(tiktoken.get_encoding)