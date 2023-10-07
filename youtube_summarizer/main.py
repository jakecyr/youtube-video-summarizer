from argparse import ArgumentParser, Namespace
import asyncio
import os

import tiktoken
from youtube_summarizer.clients.openai_client import OpenAIClient

from youtube_summarizer.youtube_video_summarizer import YouTubeVideoSummarizer
from youtube_summarizer.youtube_video import YouTubeVideo


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--video-url",
        "-v",
        help="The URL of the video to summarize.",
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

    args: Namespace = parser.parse_args()
    openai_api_key: str | None = args.openai_api_key

    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")

    if openai_api_key is None:
        raise ValueError(
            "Expected api_key parameter or OPENAI_API_KEY env var to be set.",
        )

    openai_client = OpenAIClient(openai_api_key)
    tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    summarizer = YouTubeVideoSummarizer(openai_client=openai_client, encoding=tokenizer)
    youtube_video = YouTubeVideo(args.video_url)

    if args.run_async:
        summary: str = asyncio.run(summarizer.summarize_async(youtube_video))
    else:
        summary = summarizer.summarize(youtube_video)

    print(summary)


if __name__ == "__main__":
    main()
