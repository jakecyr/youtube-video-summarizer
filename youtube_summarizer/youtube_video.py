from youtube_summarizer.url_parser import URLParser


class YouTubeVideo:

    """Represents a YouTube video."""

    def __init__(self, video_url_or_id: str) -> None:
        """Initialize the YouTubeVideo instance.

        Args:
        ----
            video_url_or_id: The URL or ID of the video.

        """
        if not video_url_or_id:
            raise ValueError("The video URL or ID is empty.")

        if not isinstance(video_url_or_id, str):
            raise TypeError("Expected string value for video URL or ID.")

        if not video_url_or_id.startswith("https"):
            self.id: str = video_url_or_id
        else:
            parsed_url = URLParser(url=video_url_or_id)
            query_string_dict: dict = parsed_url.query_string_dict()

            if "v" not in query_string_dict:
                raise ValueError(
                    "The video URL is malformed. Expected 'v' in query string.",
                )

            self.id = query_string_dict["v"]
