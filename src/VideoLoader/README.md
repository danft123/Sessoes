# VideoLoader

Loads video files from OBS recordings and extracts audio tracks for processing.

## Overview

This module provides functionality to load video files (mp4, mkv) from a directory and extract audio for downstream processing in the pipeline.

## Usage

```python
from src.VideoLoader.loader import VideoLoader

# Initialize with directory containing video files
loader = VideoLoader("data/Raw/Videos")

# Load the heaviest (largest) video file in the directory
video = loader.load_heaviest_video()

# Load a specific video file
video = loader.load_video("path/to/video.mp4")
```

## Classes

### VideoLoader

**`__init__(path_directory: str)`**
- Initializes the VideoLoader with a directory containing video files.
- Scans the directory for `.mp4` and `.mkv` files.

**`load_video(video_path: str)`**
- Loads a specific video file using OpenCV.
- Returns a VideoCapture object.
- Raises `ValueError` if the video cannot be opened.

**`load_heaviest_video()`**
- Loads the largest video file in the directory by file size.
- Useful when selecting the main recording from multiple OBS recordings.

## Dependencies

- `cv2` (OpenCV) - Video capture and frame extraction
- `ffmpeg-python` - Audio extraction from video
- `numpy` - Numerical operations

## Notes

- The current implementation returns `...` (TODO) for `load_video()`. Full frame/audio extraction is in development.
- This module is typically used at the start of the pipeline to extract audio from OBS recordings.