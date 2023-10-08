# YouTube Video Summarizer

Summarize YouTube videos using Python and GPT models.

## Installation

1. Make sure a valid version of Python is installed.
1. Make sure poetry is install `pip install poetry`.
1. Run `poetry install` to install dependencies.

## Usage

Summarize a YouTube video by copying the URL or video ID and then running the following command:

```bash
python youtube_summarizer/main.py -v GpqAQxH1Afc
```

Replacing `GpqAQxH1Afc` with the video ID of the video you want to summarize.

You can also add a `--run-async` or `-a` flag to run the code asynchronously which will speed up the execution.

## Tips

- Try changing the model to gpt-4 by specifying the `-m` flag.
- Decrease the context length with the `-c` flag or add the `-d` (detailed) flag to generate more notes.
