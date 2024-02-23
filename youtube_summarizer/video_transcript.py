from collections.abc import Generator

from loguru import logger

from youtube_summarizer.utils.tokenizer import Tokenizer


class VideoTranscript:

    """Represents a video transcript."""

    def __init__(self, transcript_chunks: list[str]) -> None:
        """Initialize the Transcript instance.

        Args:
        ----
            transcript_chunks: The list of transcript chunks.

        """
        logger.debug(
            f"Initialized transcript instance with {len(transcript_chunks)} lines.",
        )
        self._transcript_chunks: list[str] = transcript_chunks

    def get_chunks(
        self,
        token_limit: int,
        tokenizer: Tokenizer,
    ) -> Generator[str, None, None]:
        """Return a generator of transcript chunks of a given token limit.

        Args:
        ----
            token_limit: The maximum number of tokens in each chunk.
            tokenizer: The tokenizer to use.

        Returns:
        -------
            A generator of transcript chunks.

        """
        logger.debug(
            f"Converting transcript into chunks of {token_limit} max tokens...",
        )
        transcript_str: str = ""

        for i, chunk in enumerate(self._transcript_chunks):
            formatted_chunk: str = chunk if i == 0 else " " + chunk
            tokens: int = tokenizer.count_tokens(transcript_str + formatted_chunk)

            if tokens >= token_limit:
                logger.debug(f"Adding chunk: {formatted_chunk}")
                yield transcript_str
                transcript_str = ""

            transcript_str += formatted_chunk

        if transcript_str:
            logger.debug(f"Adding last chunk: {transcript_str}")
            yield transcript_str
            transcript_str = ""
