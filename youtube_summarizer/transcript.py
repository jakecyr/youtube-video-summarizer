from typing import Generator

from youtube_summarizer.tokenizer import Tokenizer
from loguru import logger


class Transcript:
    def __init__(self, transcript_chunks: list[str]) -> None:
        logger.debug(
            f"Initialized transcript instance with {len(transcript_chunks)} lines."
        )
        self._transcript_chunks: list[str] = transcript_chunks

    def get_chunks(
        self,
        token_limit: int,
        tokenizer: Tokenizer,
    ) -> Generator[str, None, None]:
        logger.debug(
            f"Converting transcript into chunks of {token_limit} max tokens..."
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
