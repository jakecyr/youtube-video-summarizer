from typing import Generator
from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptClient:
    @staticmethod
    def get_transcript(video_id: str) -> Generator[str, None, None]:
        """Gets the transcript of a video.

        Args:
            video_id: The ID of the video.

        Returns:
            The transcript of the video.
        """
        transcript: dict = YouTubeTranscriptApi.get_transcript(video_id)

        for line in transcript:
            yield line["text"]
