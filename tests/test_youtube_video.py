import pytest
from youtube_summarizer.youtube_video import YouTubeVideo


def test_parses_id_from_url() -> None:
    url = "https://www.youtube.com/watch?v=12345"
    parser = YouTubeVideo(url)
    assert parser.id == "12345"


def test_parses_id_from_id() -> None:
    url = "12345"
    parser = YouTubeVideo(url)
    assert parser.id == "12345"


def test_empty_str_raises_value_error() -> None:
    url: str = ""
    with pytest.raises(ValueError, match="The video URL or ID is empty"):
        parser = YouTubeVideo(url)


def test_none_raises_value_error() -> None:
    url: str | None = None
    with pytest.raises(ValueError, match="The video URL or ID is empty"):
        parser = YouTubeVideo(url)  # type: ignore


def test_url_without_query_string_raises_value_error() -> None:
    url = "https://www.youtube.com/watch"
    with pytest.raises(ValueError, match="The video URL is malformed"):
        YouTubeVideo(url)


def test_url_without_video_id_in_query_string_raises_value_error() -> None:
    url = "https://www.youtube.com/watch?referrer=jake"
    with pytest.raises(ValueError, match="The video URL is malformed"):
        YouTubeVideo(url)


def test_non_string_value_raises_type_error() -> None:
    url = 12345
    with pytest.raises(TypeError, match="Expected string value for video URL or ID"):
        YouTubeVideo(url)  # type: ignore
