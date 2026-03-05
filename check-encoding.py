path = r"C:\eyal\code\LLM\sample.csv"
with open(path, "rb") as f:
    first = f.read(8)
print(first)                        # show the raw bytes at the start
import chardet
print(chardet.detect(first + f.read(4096)))