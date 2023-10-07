from typing import Generator
import tiktoken
from youtube_summarizer.clients.openai_client import OpenAIClient
from youtube_summarizer.clients.youtube_transcript_client import YouTubeTranscriptClient
from youtube_summarizer.youtube_video import YouTubeVideo
import asyncio

GPT_35_TURBO_TOKEN_LIMIT = 4096

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a YouTube summarizer bot. Summarize the provided content into "
    "a few concise bullet points capturing important ideas."
)


class YouTubeVideoSummarizer:
    def __init__(
        self, openai_client: OpenAIClient, encoding: tiktoken.Encoding
    ) -> None:
        self._openai_client: OpenAIClient = openai_client
        self._encoding: tiktoken.Encoding = encoding

    def summarize(
        self,
        youtube_video: YouTubeVideo,
        *,
        model="gpt-3.5-turbo-0613",
        token_limit=GPT_35_TURBO_TOKEN_LIMIT
    ) -> str:
        """Summarize a YouTube video.

        Args:
          video_url: The URL of the video to summarize.
          model: The chat completion model to use for summarization.
          token_limit: The token limit of the model being used for summarization.

        Returns:
          A summary of the video.
        """
        transcript: Generator[str, None, None] = YouTubeTranscriptClient.get_transcript(
            video_id=youtube_video.id
        )
        transcript_str: str = ""
        summary: str = ""

        for line in transcript:
            if len(transcript_str + line) >= token_limit:
                summary += self._summarize_chunk(transcript_str, model)
                transcript_str = ""

            transcript_str += line

        if transcript_str:
            summary += self._summarize_chunk(transcript_str, model)

        return summary

    async def summarize_async(
        self,
        youtube_video: YouTubeVideo,
        *,
        model="gpt-3.5-turbo-0613",
        token_limit=GPT_35_TURBO_TOKEN_LIMIT
    ) -> str:
        """Summarize a YouTube video.

        Args:
          video_url: The URL of the video to summarize.
          model: The chat completion model to use for summarization.
          token_limit: The token limit of the model being used for summarization.

        Returns:
          A summary of the video.
        """
        transcript: Generator[str, None, None] = YouTubeTranscriptClient.get_transcript(
            video_id=youtube_video.id
        )
        transcript_str: str = ""
        tasks: list[asyncio.Task] = []

        for line in transcript:
            if len(transcript_str + line) >= token_limit:
                tasks.append(
                    asyncio.ensure_future(
                        self._summarize_chunk_async(transcript_str, model)
                    )
                )
                transcript_str = ""

            transcript_str += line

        if transcript_str:
            tasks.append(
                asyncio.ensure_future(
                    self._summarize_chunk_async(transcript_str, model)
                )
            )

        task_results: list[str] = await asyncio.gather(*tasks)
        return "\n".join(task_results)

    def _summarize_chunk(self, chunk: str, model: str) -> str:
        """Summarize a chunk of text.

        Args:
          chunk: The chunk of text to summarize.

        Returns:
          The summarized chunk.
        """
        response: dict = self._openai_client.generate_chat_completion(
            user_prompt=chunk,
            model=model,
            temperature=0.1,
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        )

        return response["content"]

    async def _summarize_chunk_async(self, chunk: str, model: str) -> str:
        """Summarize a chunk of text.

        Args:
          chunk: The chunk of text to summarize.

        Returns:
          The summarized chunk.
        """
        response: dict = await self._openai_client.generate_chat_completion_async(
            user_prompt=chunk,
            model=model,
            temperature=0.1,
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        )

        return response["content"]
