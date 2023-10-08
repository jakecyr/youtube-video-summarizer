from youtube_transcript_api import YouTubeTranscriptApi

from youtube_summarizer.transcript import Transcript
from loguru import logger


class YouTubeTranscriptClient:
    @staticmethod
    def get_transcript(video_id: str) -> Transcript:
        """Gets the transcript of a video.

        Args:
            video_id: The ID of the video.

        Returns:
            The transcript of the video.
        """
        transcript: dict = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_chunks: list[str] = []

        logger.debug(
            f"Received YouTube transcript for video {video_id} with {len(transcript)} lines."
        )

        for line in transcript:
            transcript_chunks.append(line["text"])

        return Transcript(transcript_chunks)
