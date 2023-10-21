import asyncio
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import tiktoken
from loguru import logger

from youtube_summarizer.clients.openai_client import OpenAIClient
from youtube_summarizer.clients.youtube_transcript_client import YouTubeTranscriptClient
from youtube_summarizer.tokenizer import Tokenizer
from youtube_summarizer.youtube_video import YouTubeVideo

if TYPE_CHECKING:
    from collections.abc import Generator

    from youtube_summarizer.transcript import Transcript

GPT_35_TURBO_TOKEN_LIMIT = 4096

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a YouTube summarizer bot. Summarize the provided content into "
    "a few concise bullet points capturing important ideas."
)

DETAILED_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a YouTube summarizer bot. Summarize the provided content into "
    "detailed bullet points fully capturing important ideas."
)


class VideoSummarizationMeta(NamedTuple):

    """Meta information about the video summary."""

    prompt_tokens: int
    completion_tokens: int


class VideoSummarizationBulletedList(NamedTuple):

    """A string summary of a YouTube video as bulleted list."""

    video_id: str
    summary: str
    meta: VideoSummarizationMeta


class VideoSummarizationList(NamedTuple):

    """A listed summary of a YouTube video as Python list of strings."""

    video_id: str
    summary: list[str]
    meta: VideoSummarizationMeta


class SummarizationOutputFormat(str, Enum):

    """The format to return the summary in."""

    BULLETED_LIST = "bulleted_list"
    LIST = "list"


class YouTubeVideoSummarizer:

    """YouTube Video Summarizer."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        *,
        model_name="gpt-3.5-turbo",
        token_limit=GPT_35_TURBO_TOKEN_LIMIT,
        detailed_summary: bool = False,
    ) -> None:
        """Initialize the YouTubeVideoSummarizer instance.

        Args:
        ----
          openai_client: The OpenAI API client.
          model_name: The chat completion model to use for summarization.
          token_limit: The maximum number of tokens to use for summarization.
          detailed_summary: Whether to generate a detailed or short summary.
        """
        self._openai_client: OpenAIClient = openai_client
        self._tokenizer = Tokenizer(tiktoken.encoding_for_model(model_name))
        self._model_name: str = model_name
        self._token_limit: int = token_limit
        self._system_prompt: str = (
            DETAILED_SUMMARIZATION_SYSTEM_PROMPT
            if detailed_summary
            else SUMMARIZATION_SYSTEM_PROMPT
        )

    def summarize(
        self,
        youtube_video: YouTubeVideo,
        *,
        output_format: SummarizationOutputFormat | str = SummarizationOutputFormat.LIST,
        temperature: float = 0.1,
    ) -> VideoSummarizationBulletedList | VideoSummarizationList:
        """Summarize a YouTube video.

        Args:
        ----
          youtube_video: The URL of the video to summarize.
          output_format: The format to return the summary in.
          temperature: The temperature to use for the model.

        Returns:
        -------
          An object containing the summary and meta information.

        Raises:
        ------
          ValueError: If the output format is invalid.
        """
        if (
            output_format
            not in SummarizationOutputFormat._value2member_map_  # noqa: SLF001
        ):
            raise ValueError(f"Invalid output format: {output_format}.")

        logger.debug(f"Summarizing video {youtube_video.id}...")

        transcript: Transcript = YouTubeTranscriptClient.get_transcript(
            video_id=youtube_video.id,
        )
        transcript_chunks: Generator[str, None, None] = transcript.get_chunks(
            self._token_limit,
            self._tokenizer,
        )
        summaries: list[str] = []
        all_chunks: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in transcript_chunks:
            summary, usage = self._summarize_chunk(
                chunk,
                self._model_name,
                temperature=temperature,
            )
            logger.debug(f"Summarized chunk: {summary}")
            logger.debug(f"Usage: {usage}")
            summaries.append(summary)
            all_chunks.append(chunk)
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]

        meta_information = VideoSummarizationMeta(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return self._get_formatted_summarization(
            video_id=youtube_video.id,
            meta_information=meta_information,
            output_format=output_format,
            summary_chunks=summaries,
        )

    async def summarize_async(
        self,
        youtube_video: YouTubeVideo,
        *,
        output_format: SummarizationOutputFormat | str = SummarizationOutputFormat.LIST,
        temperature: float = 0.1,
    ) -> VideoSummarizationBulletedList | VideoSummarizationList:
        """Summarize a YouTube video.

        Args:
        ----
          youtube_video: The URL of the video to summarize.
          output_format: The format to return the summary in.
          temperature: The temperature to use for the model.

        Returns:
        -------
          An object containing the summary and meta information.

        Raises:
        ------
          ValueError: If the output format is invalid.
        """
        if (
            output_format
            not in SummarizationOutputFormat._value2member_map_  # noqa: SLF001
        ):
            raise ValueError(f"Invalid output format: {output_format}.")

        logger.debug(f"Summarizing video {youtube_video.id}...")

        transcript: Transcript = YouTubeTranscriptClient.get_transcript(
            video_id=youtube_video.id,
        )
        transcript_chunks: Generator[str, None, None] = transcript.get_chunks(
            self._token_limit,
            self._tokenizer,
        )
        tasks: list[asyncio.Task] = []

        for chunk in transcript_chunks:
            tasks.append(
                asyncio.create_task(
                    self._summarize_chunk_async(
                        chunk,
                        self._model_name,
                        temperature=temperature,
                    ),
                ),
            )

        results: list[tuple[str, dict]] = await asyncio.gather(*tasks)
        prompt_tokens = 0
        completion_tokens = 0
        summary_chunks: list[str] = []

        for result in results:
            summaries, usage_dict = result
            prompt_tokens += usage_dict["prompt_tokens"]
            completion_tokens += usage_dict["completion_tokens"]
            logger.debug(f'summary: "{summaries}"')
            summary_chunks.append(summaries)

        meta_information = VideoSummarizationMeta(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return self._get_formatted_summarization(
            video_id=youtube_video.id,
            summary_chunks=summary_chunks,
            meta_information=meta_information,
            output_format=output_format,
        )

    def _get_formatted_summarization(
        self,
        video_id: str,
        summary_chunks: list[str],
        meta_information: VideoSummarizationMeta,
        output_format: SummarizationOutputFormat | str,
    ) -> VideoSummarizationBulletedList | VideoSummarizationList:
        if output_format == SummarizationOutputFormat.BULLETED_LIST:
            summary_str: str = "\n".join(summary_chunks)

            return VideoSummarizationBulletedList(
                video_id=video_id,
                summary=summary_str,
                meta=meta_information,
            )

        summary_list: list[str] = []

        for chunk in summary_chunks:
            summary_list.extend(
                [
                    chunk[2:] if chunk.startswith("- ") else chunk
                    for chunk in chunk.split("\n")
                    if chunk
                ],
            )

        return VideoSummarizationList(
            video_id=video_id,
            summary=summary_list,
            meta=meta_information,
        )

    def _summarize_chunk(
        self,
        chunk: str,
        model: str,
        *,
        temperature: float = 0.1,
    ) -> tuple[str, dict]:
        """Summarize a chunk of text.

        Args:
        ----
          chunk: The chunk of text to summarize.
          model: The model to use for the API.
          temperature: The temperature to use for the model.

        Returns:
        -------
          The summarized chunk and usage.
        """
        logger.debug(f"Summarizing chunk with model {model}...")

        response, usage = self._openai_client.generate_chat_completion(
            user_prompt=chunk,
            model=model,
            temperature=temperature,
            system_prompt=self._system_prompt,
        )

        return response["content"], usage

    async def _summarize_chunk_async(
        self,
        chunk: str,
        model: str,
        *,
        temperature: float = 0.1,
    ) -> tuple[str, dict]:
        """Summarize a chunk of text.

        Args:
        ----
          chunk: The chunk of text to summarize.
          model: The model to use for the API.
          temperature: The temperature to use for the model.

        Returns:
        -------
          The summarized chunk and usage.
        """
        logger.debug(f"Summarizing chunk async with model {model}...")

        response, usage = await self._openai_client.generate_chat_completion_async(
            user_prompt=chunk,
            model=model,
            temperature=temperature,
            system_prompt=self._system_prompt,
        )

        return response["content"], usage
