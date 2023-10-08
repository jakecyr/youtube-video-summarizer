import tiktoken


class Tokenizer:
    def __init__(self, encoding: tiktoken.Encoding) -> None:
        self._encoding: tiktoken.Encoding = encoding

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, encodings: list[int]) -> str:
        return self._encoding.decode(encodings)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))
