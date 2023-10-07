from youtube_summarizer.url_parser import URLParser


class YouTubeVideo:
    def __init__(self, video_url: str) -> None:
        parsed_url = URLParser(url=video_url)
        query_string_dict: dict = parsed_url.query_string_dict()

        if "v" not in query_string_dict:
            raise ValueError(
                "The video URL is malformed. Expected 'v' in query string."
            )

        self.id = query_string_dict["v"]
