import urllib.request
import os

class DummyTextLoader:
    def __init__(self, source: str, is_url: bool = False):
        if is_url:
            file_path = os.path.basename(source)
            urllib.request.urlretrieve(source, file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                self.raw_text = f.read()
        else:
            if os.path.exists(source):
                with open(source, "r", encoding="utf-8") as f:
                    self.raw_text = f.read()
            else:
                # Treat as direct string input
                self.raw_text = source

    def print_text(self):
        print(self.raw_text)