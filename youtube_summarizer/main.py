from argparse import ArgumentParser, Namespace
import asyncio
import os
import sys

from youtube_summarizer.clients.openai_client import OpenAIClient

from youtube_summarizer.youtube_video_summarizer import YouTubeVideoSummarizer
from youtube_summarizer.youtube_video import YouTubeVideo
from loguru import logger

# Add a new logging handler.
logger.remove()
logger.add(sys.stdout, level="INFO")


def main() -> None:
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
        help="The OpenAI Chat Completion model context length. Should correspond to the model name.",
        default=4096,
        type=int,
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
        summary: str = asyncio.run(summarizer.summarize_async(youtube_video))
    else:
        summary = summarizer.summarize(youtube_video)

    logger.info(f"Summarized video.")
    logger.info(summary)


if __name__ == "__main__":
    main()
