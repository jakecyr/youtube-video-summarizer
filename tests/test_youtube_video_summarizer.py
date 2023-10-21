from unittest import mock

import pytest

from youtube_summarizer.youtube_video import YouTubeVideo
from youtube_summarizer.youtube_video_summarizer import (
    VideoSummarizationBulletedList,
    VideoSummarizationList,
    YouTubeVideoSummarizer,
)

# Mock the YouTubeVideo class.
mocked_youtube_video = YouTubeVideo(video_url_or_id="test_video_id")

mock_summary: str = """
- This is a summary
- This is another summary
- This is a third summary
"""

mock_summary_list: list[str] = [
    "This is a summary",
    "This is another summary",
    "This is a third summary",
]

# Mock response and usage data.
mocked_response: dict[str, str] = {"content": "This is a summarized content"}
mocked_usage: dict[str, int] = {"prompt_tokens": 10, "completion_tokens": 20}


@pytest.fixture()
def open_ai_client() -> mock.MagicMock:
    return mock.MagicMock()


@pytest.fixture()
def youtube_video_summarizer(open_ai_client: mock.MagicMock) -> YouTubeVideoSummarizer:
    return YouTubeVideoSummarizer(open_ai_client)


def test_init_does_not_throw(youtube_video_summarizer: YouTubeVideoSummarizer) -> None:
    assert youtube_video_summarizer


def test_summarize_returns_tuple() -> None:
    with mock.patch(
        "youtube_summarizer.clients.youtube_transcript_client.YouTubeTranscriptClient.get_transcript",
    ) as mock_transcript, mock.patch(
        "youtube_summarizer.youtube_video_summarizer.YouTubeVideoSummarizer._summarize_chunk",
    ) as mock_chunk:
        # Mock transcript response and chunk summarization
        mock_transcript.return_value = mock.Mock(
            get_chunks=mock.Mock(return_value=["chunk1", "chunk2"]),
        )
        mock_chunk.return_value = (mock_summary, mocked_usage)
        client = mock.Mock()
        summarizer = YouTubeVideoSummarizer(openai_client=client)
        summarization = summarizer.summarize(mocked_youtube_video)

        assert isinstance(summarization, tuple)


def test_summarize_returns_tuple_of_expected_values() -> None:
    mock_chunks: list[str] = ["chunk1", "chunk2"]

    with mock.patch(
        "youtube_summarizer.clients.youtube_transcript_client.YouTubeTranscriptClient.get_transcript",
    ) as mock_transcript, mock.patch(
        "youtube_summarizer.youtube_video_summarizer.YouTubeVideoSummarizer._summarize_chunk",
    ) as mock_chunk:
        # Mock transcript response and chunk summarization
        mock_transcript.return_value = mock.Mock(
            get_chunks=mock.Mock(return_value=mock_chunks),
        )
        mock_chunk.return_value = (mock_summary, mocked_usage)
        client = mock.Mock()
        summarizer = YouTubeVideoSummarizer(openai_client=client)
        summarization = summarizer.summarize(mocked_youtube_video)

    num_chunks: int = len(mock_chunks)

    assert summarization.summary == (mock_summary_list * num_chunks)
    assert summarization.meta.prompt_tokens == (
        mocked_usage["prompt_tokens"] * num_chunks
    )
    assert summarization.meta.completion_tokens == (
        mocked_usage["completion_tokens"] * num_chunks
    )


def test_summarize_with_no_format_returns_list_object() -> None:
    mock_chunks: list[str] = ["chunk1", "chunk2"]

    with mock.patch(
        "youtube_summarizer.clients.youtube_transcript_client.YouTubeTranscriptClient.get_transcript",
    ) as mock_transcript, mock.patch(
        "youtube_summarizer.youtube_video_summarizer.YouTubeVideoSummarizer._summarize_chunk",
    ) as mock_chunk:
        # Mock transcript response and chunk summarization
        mock_transcript.return_value = mock.Mock(
            get_chunks=mock.Mock(return_value=mock_chunks),
        )
        mock_chunk.return_value = (mock_summary, mocked_usage)
        client = mock.Mock()
        summarizer = YouTubeVideoSummarizer(openai_client=client)
        summarization = summarizer.summarize(mocked_youtube_video)

    assert isinstance(summarization, VideoSummarizationList)


def test_summarize_with_bulleted_list_format_returns_expected_str_summary() -> None:
    mock_chunks: list[str] = ["chunk1", "chunk2"]

    with mock.patch(
        "youtube_summarizer.clients.youtube_transcript_client.YouTubeTranscriptClient.get_transcript",
    ) as mock_transcript, mock.patch(
        "youtube_summarizer.youtube_video_summarizer.YouTubeVideoSummarizer._summarize_chunk",
    ) as mock_chunk:
        # Mock transcript response and chunk summarization
        mock_transcript.return_value = mock.Mock(
            get_chunks=mock.Mock(return_value=mock_chunks),
        )
        mock_chunk.return_value = (mock_summary, mocked_usage)
        client = mock.Mock()
        summarizer = YouTubeVideoSummarizer(openai_client=client)
        summarization = summarizer.summarize(
            mocked_youtube_video,
            output_format="bulleted_list",
        )

    num_chunks: int = len(mock_chunks)

    assert summarization.summary == (mock_summary + "\n" + mock_summary)
    assert summarization.meta.prompt_tokens == (
        mocked_usage["prompt_tokens"] * num_chunks
    )
    assert summarization.meta.completion_tokens == (
        mocked_usage["completion_tokens"] * num_chunks
    )


def test_summarize_with_bulleted_list_format_returns_expected_object() -> None:
    mock_chunks: list[str] = ["chunk1", "chunk2"]

    with mock.patch(
        "youtube_summarizer.clients.youtube_transcript_client.YouTubeTranscriptClient.get_transcript",
    ) as mock_transcript, mock.patch(
        "youtube_summarizer.youtube_video_summarizer.YouTubeVideoSummarizer._summarize_chunk",
    ) as mock_chunk:
        # Mock transcript response and chunk summarization
        mock_transcript.return_value = mock.Mock(
            get_chunks=mock.Mock(return_value=mock_chunks),
        )
        mock_chunk.return_value = (mock_summary, mocked_usage)
        client = mock.Mock()
        summarizer = YouTubeVideoSummarizer(openai_client=client)
        summarization = summarizer.summarize(
            mocked_youtube_video,
            output_format="bulleted_list",
        )

    assert isinstance(summarization, VideoSummarizationBulletedList)


def test_summarize_with_invalid_output_format_raises_value_error(
    youtube_video_summarizer: YouTubeVideoSummarizer,
) -> None:
    with pytest.raises(ValueError, match="Invalid output format"):
        youtube_video_summarizer.summarize(mocked_youtube_video, output_format="bad")
