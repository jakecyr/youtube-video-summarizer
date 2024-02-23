import asyncio
import os
import sys
from argparse import ArgumentParser, Namespace
from typing import TYPE_CHECKING

from loguru import logger

from youtube_summarizer.clients.openai_client import OpenAIClient
from youtube_summarizer.youtube_video import YouTubeVideo
from youtube_summarizer.youtube_video_summarizer import (
    SummarizationOutputFormat,
    YouTubeVideoSummarizer,
)

if TYPE_CHECKING:
    from youtube_summarizer.video_usage_meta import VideoUsageMeta

# Add a new logging handler.
logger.remove()
logger.add(sys.stdout, level=os.environ.get("LOG_LEVEL", "INFO").upper())


def main() -> None:
    """Run the CLI."""
    parser = ArgumentParser()
    parser.add_argument(
        "--video-url-or-id",
        "-v",
        help="The URL or ID of the video to summarize.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--openai-api-key",
        "-k",
        help="The OpenAI API key.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--run-async",
        "-a",
        help="If should run asynchronously.",
        action="store_true",
    )
    parser.add_argument(
        "--detailed",
        "-d",
        help="If a detailed summary should be generated.",
        action="store_true",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        help="The OpenAI Chat Completion model name to use.",
        default="gpt-3.5-turbo",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--model-context-length",
        "-c",
        help=(
            "The OpenAI Chat Completion model context length. "
            "Should correspond to the model name."
        ),
        default=4096,
        type=int,
        required=False,
    )
    parser.add_argument(
        "--output-format",
        "-o",
        help="The output format of the summary.",
        default=SummarizationOutputFormat.LIST.value,
        type=str,
        required=False,
    )

    args: Namespace = parser.parse_args()
    openai_api_key: str | None = args.openai_api_key

    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openai_api_key is None:
        raise ValueError(
            "Expected api_key parameter or OPENAI_API_KEY env var to be set.",
        )

    openai_client = OpenAIClient(openai_api_key)
    summarizer = YouTubeVideoSummarizer(
        openai_client=openai_client,
        model_name=args.model_name,
        token_limit=args.model_context_length,
        detailed_summary=args.detailed,
    )
    youtube_video = YouTubeVideo(args.video_url_or_id)

    logger.info(f"Summarizing video {args.video_url_or_id}...")

    if args.run_async:
        summarization = asyncio.run(
            summarizer.summarize_async(youtube_video, output_format=args.output_format),
        )
    else:
        summarization = summarizer.summarize(
            youtube_video,
            output_format=args.output_format,
        )

    meta: VideoUsageMeta = summarization.meta
    prompt_tokens: int = meta.prompt_tokens
    completion_tokens: int = meta.completion_tokens
    total_tokens: int = prompt_tokens + completion_tokens

    logger.info(f"Summarized video: {summarization.summary}")
    logger.info(f"Prompt tokens: {prompt_tokens}")
    logger.info(f"Completion tokens: {completion_tokens}")
    logger.info(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
