from typing import TYPE_CHECKING

import tiktoken

from youtube_summarizer.video_transcript import VideoTranscript
from youtube_summarizer.utils.tokenizer import Tokenizer

if TYPE_CHECKING:
    from collections.abc import Generator


def test_get_chunks_returns_expected_chunk_length() -> None:
    transcript_strs: list[str] = ["Hello", "world"]
    transcript = VideoTranscript(transcript_strs)
    encoding: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = Tokenizer(encoding)
    chunks: Generator[str, None, None] = transcript.get_chunks(2, tokenizer=tokenizer)
    chunks_list = list(chunks)

    assert len(chunks_list) == len(transcript_strs)


def test_get_chunks_returns_expected_chunk_content() -> None:
    transcript_strs: list[str] = ["Hello", "world"]
    transcript = VideoTranscript(transcript_strs)
    encoding: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = Tokenizer(encoding)
    chunks: Generator[str, None, None] = transcript.get_chunks(2, tokenizer=tokenizer)
    chunks_list = list(chunks)

    assert "".join(chunks_list) == " ".join(transcript_strs)


def test_get_chunks_with_high_context_length_returns_one_chunk() -> None:
    transcript_strs: list[str] = ["Hello", "world"]
    transcript = VideoTranscript(transcript_strs)
    encoding: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = Tokenizer(encoding)
    chunks: Generator[str, None, None] = transcript.get_chunks(10, tokenizer=tokenizer)
    chunks_list = list(chunks)

    assert len(chunks_list) == 1


def test_get_chunks_with_high_context_length_returns_one_chunk_with_all_content() -> (
    None
):
    transcript_strs: list[str] = ["Hello", "world"]
    transcript = VideoTranscript(transcript_strs)
    encoding: tiktoken.Encoding = tiktoken.get_encoding("cl100k_base")
    tokenizer = Tokenizer(encoding)
    chunks: Generator[str, None, None] = transcript.get_chunks(10, tokenizer=tokenizer)
    chunks_list = list(chunks)

    assert " ".join(chunks_list) == " ".join(transcript_strs)
