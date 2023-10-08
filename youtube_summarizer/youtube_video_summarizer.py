from typing import Generator
import tiktoken
from youtube_summarizer.clients.openai_client import OpenAIClient
from youtube_summarizer.clients.youtube_transcript_client import YouTubeTranscriptClient
from youtube_summarizer.tokenizer import Tokenizer
from youtube_summarizer.transcript import Transcript
from youtube_summarizer.youtube_video import YouTubeVideo
import asyncio
from loguru import logger

GPT_35_TURBO_TOKEN_LIMIT = 4096

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a YouTube summarizer bot. Summarize the provided content into "
    "a few concise bullet points capturing important ideas."
)

DETAILED_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a YouTube summarizer bot. Summarize the provided content into "
    "detailed bullet points fully capturing important ideas."
)


class YouTubeVideoSummarizer:
    def __init__(
        self,
        openai_client: OpenAIClient,
        *,
        model_name="gpt-3.5-turbo",
        token_limit=GPT_35_TURBO_TOKEN_LIMIT,
        detailed_summary: bool = False,
        temperature: float = 0.1,
    ) -> None:
        """Initializes the YouTubeVideoSummarizer instance.

        Args:
          openai_client: The OpenAI API client.
          model_name: The chat completion model to use for summarization.
          token_limit: The maximum number of tokens to use for summarization.
          detailed_summary: Whether to generate a detailed or short summary.
          temperature: The temperature to use for the model. Changes the creativity of the model.
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
        self._temperature: float = temperature

    def summarize(self, youtube_video: YouTubeVideo) -> str:
        """Summarize a YouTube video.

        Args:
          video_url: The URL of the video to summarize.

        Returns:
          A summary of the video.
        """
        logger.debug(f"Summarizing video {youtube_video.id}...")

        transcript: Transcript = YouTubeTranscriptClient.get_transcript(
            video_id=youtube_video.id
        )
        transcript_chunks: Generator[str, None, None] = transcript.get_chunks(
            self._token_limit, self._tokenizer
        )
        summaries: list[str] = []
        all_chunks: list[str] = []

        for chunk in transcript_chunks:
            summary: str = self._summarize_chunk(chunk, self._model_name)
            logger.debug(f"Summarized chunk: {summary}")
            summaries.append(summary)
            all_chunks.append(chunk)

        return "\n".join(summaries)

    async def summarize_async(self, youtube_video: YouTubeVideo) -> str:
        """Summarize a YouTube video.

        Args:
          video_url: The URL of the video to summarize.

        Returns:
          A summary of the video.
        """
        logger.debug(f"Summarizing video {youtube_video.id}...")

        transcript: Transcript = YouTubeTranscriptClient.get_transcript(
            video_id=youtube_video.id
        )
        transcript_chunks: Generator[str, None, None] = transcript.get_chunks(
            self._token_limit, self._tokenizer
        )
        tasks: list[asyncio.Task] = []

        for chunk in transcript_chunks:
            tasks.append(
                asyncio.create_task(
                    self._summarize_chunk_async(chunk, self._model_name)
                )
            )

        summaries: list[str] = await asyncio.gather(*tasks)

        return "\n".join(summaries)

    def _summarize_chunk(self, chunk: str, model: str) -> str:
        """Summarize a chunk of text.

        Args:
          chunk: The chunk of text to summarize.

        Returns:
          The summarized chunk.
        """
        logger.debug(f"Summarizing chunk with model {model}...")

        response: dict = self._openai_client.generate_chat_completion(
            user_prompt=chunk,
            model=model,
            temperature=self._temperature,
            system_prompt=self._system_prompt,
        )

        return response["content"]

    async def _summarize_chunk_async(self, chunk: str, model: str) -> str:
        """Summarize a chunk of text.

        Args:
          chunk: The chunk of text to summarize.

        Returns:
          The summarized chunk.
        """
        logger.debug(f"Summarizing chunk async with model {model}...")

        response: dict = await self._openai_client.generate_chat_completion_async(
            user_prompt=chunk,
            model=model,
            temperature=self._temperature,
            system_prompt=self._system_prompt,
        )

        return response["content"]
