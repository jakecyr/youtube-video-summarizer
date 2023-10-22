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

ANSWER_NOT_FOUND = "ANSWER_NOT_FOUND"

QA_SYSTEM_PROMPT: str = (
    "You are a YouTube question answer bot. Given a YouTube video transcript chunk "
    f"either return the answer to the question, or {ANSWER_NOT_FOUND} if the answer "
    "cannot be found in the provided context."
)

QA_CHUNK_PROMPT: str = """
Video transcript chunk:
{chunk}

Question:
{question}
"""


class VideoSummarizationMeta(NamedTuple):

    """Meta information about the video summary."""

    prompt_tokens: int
    completion_tokens: int


class VideoQAResponse(NamedTuple):

    """An object with the question and answer from the video."""

    video_id: str
    summary: list[str]
    question: str
    answer: str | None
    meta: VideoSummarizationMeta


class YouTubeVideoQA:

    """YouTube Video QA."""

    def __init__(
        self,
        openai_client: OpenAIClient,
        *,
        model_name="gpt-3.5-turbo",
        token_limit=GPT_35_TURBO_TOKEN_LIMIT,
    ) -> None:
        """Initialize the YouTubeVideoSummarizer instance.

        Args:
        ----
          openai_client: The OpenAI API client.
          model_name: The chat completion model to use for summarization.
          token_limit: The maximum number of tokens to use for summarization.
        """
        self._openai_client: OpenAIClient = openai_client
        self._tokenizer = Tokenizer(tiktoken.encoding_for_model(model_name))
        self._model_name: str = model_name
        self._token_limit: int = token_limit
        self._system_prompt: str = QA_SYSTEM_PROMPT

    def answer_question(
        self,
        youtube_video: YouTubeVideo,
        question: str,
        *,
        temperature: float = 0.1,
    ) -> VideoQAResponse:
        """Ask a question about a YouTube video and get a response.

        Args:
        ----
          youtube_video: The URL of the video to summarize.
          question: The question to ask.
          temperature: The temperature to use for the model.

        Returns:
        -------
          An object containing the answer and meta information.
        """
        logger.debug(f"Answering question with video {youtube_video.id}...")
        transcript: Transcript = YouTubeTranscriptClient.get_transcript(
            video_id=youtube_video.id,
        )
        transcript_chunks: Generator[str, None, None] = transcript.get_chunks(
            self._token_limit,
            self._tokenizer,
        )
        summaries: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in transcript_chunks:
            answer, usage = self._check_chunk_for_answer(
                chunk=chunk,
                question=question,
                model=self._model_name,
                temperature=temperature,
            )

            if answer:
                logger.debug(f"Found answer: {answer}")

                return VideoQAResponse(
                    video_id=youtube_video.id,
                    summary=summaries,
                    question=question,
                    answer=answer,
                    meta=VideoSummarizationMeta(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    ),
                )
            else:
                logger.debug(f"No answer found in chunk.")

            logger.debug(f"Usage: {usage}")
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]

        return VideoQAResponse(
            video_id=youtube_video.id,
            summary=summaries,
            question=question,
            answer=None,
            meta=VideoSummarizationMeta(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
        )

    def _check_chunk_for_answer(
        self,
        chunk: str,
        question: str,
        model: str,
        *,
        temperature: float = 0.1,
    ) -> tuple[str | None, dict]:
        """Answer a question about a chunk of text.

        Args:
        ----
          chunk: The chunk of text to summarize.
          question: The question to ask.
          model: The model to use for the API.
          temperature: The temperature to use for the model.

        Returns:
        -------
          The answer found or None if no answer is found.
        """
        logger.debug(f"Trying to answer question with chunk using model {model}...")

        prompt = QA_CHUNK_PROMPT.format(
            chunk=chunk,
            question=question,
        )
        response, usage = self._openai_client.generate_chat_completion(
            user_prompt=prompt,
            model=model,
            temperature=temperature,
            system_prompt=self._system_prompt,
        )

        if response["content"] == ANSWER_NOT_FOUND:
            return None, usage

        return response["content"], usage


if __name__ == "__main__":
    from dotenv import load_dotenv
    from youtube_summarizer.clients.openai_client import OpenAIClient

    load_dotenv()

    openai_client = OpenAIClient()
    video_qa = YouTubeVideoQA(openai_client=openai_client)
    answer: VideoQAResponse = video_qa.answer_question(
        YouTubeVideo("RmjHyFM525Q"), "What service does Matt use to make websites?"
    )

    print(answer)