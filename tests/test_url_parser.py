from youtube_summarizer.utils.url_parser import URLParser


def test_query_string_dict_returns_dict() -> None:
    url = "https://www.youtube.com/watch?v=12345"
    parser = URLParser(url)
    result = parser.query_string_dict()
    assert isinstance(result, dict)


def test_query_string_dict_returns_expected_values() -> None:
    url = "https://www.youtube.com/watch?v=12345"
    parser = URLParser(url)
    result: dict = parser.query_string_dict()
    assert result == {"v": "12345"}
