from loguru import logger
from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore[import]

from youtube_summarizer.transcript import Transcript


class YouTubeTranscriptClient:

    """A client for the YouTube Transcript API."""

    @staticmethod
    def get_transcript(video_id: str) -> Transcript:
        """Get the transcript of a video.

        Args:
        ----
            video_id: The ID of the video.

        Returns:
        -------
            The transcript of the video.
        """
        transcript: dict = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_chunks: list[str] = []

        logger.debug(
            f"Received YouTube transcript for video {video_id} "
            "with {len(transcript)} lines.",
        )

        for line in transcript:
            transcript_chunks.append(line["text"])

        return Transcript(transcript_chunks)
