import tiktoken


class Tokenizer:

    """Tokenizer service for encoding and decoding text."""

    def __init__(self, encoding: tiktoken.Encoding) -> None:
        """Initialize the Tokenizer instance.

        Args:
        ----
            encoding: The encoding to use.
        """
        self._encoding: tiktoken.Encoding = encoding

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of integers.

        Args:
        ----
            text: The string to encode.

        Returns:
        -------
            A list of tokens.
        """
        return self._encoding.encode(text)

    def decode(self, encodings: list[int]) -> str:
        """Decode a list of tokens into a string.

        Args:
        ----
            encodings: The list of tokens to decode.

        Returns:
        -------
            The decoded string.
        """
        return self._encoding.decode(encodings)

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in a string.

        Args:
        ----
            text: The string to count tokens in.

        Returns:
        -------
            The number of tokens.
        """
        return len(self.encode(text))
