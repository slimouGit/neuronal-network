import re
import urllib.request

# 1️⃣ Textdatei herunterladen
url = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

# 2️⃣ Datei lesen
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

print("Total number of characters:", len(raw_text))
print(raw_text[:99])  # zeigt die ersten 99 Zeichen als Vorschau

# 3️⃣ Text vorverarbeiten: Satzzeichen & Leerzeichen trennen
preprocessed = re.split(r'([,.:;?_!"()\[\]\'—-]|\s+)', raw_text)

# 4️⃣ Leere Elemente entfernen & Whitespace abschneiden
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# 5️⃣ Anzahl der Tokens ausgeben
print("Total number of tokens:", len(preprocessed))
