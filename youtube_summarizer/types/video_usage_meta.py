from typing import NamedTuple


class VideoUsageMeta(NamedTuple):

    """Meta information about the video request."""

    prompt_tokens: int
    completion_tokens: int
