# Voice.Transcript

Transcribes audio recordings using WhisperX with word-level timestamps and speaker diarization.

## Overview

This module provides high-level transcription capabilities using [WhisperX](https://github.com/m-bain/WhisperX), which combines OpenAI's Whisper model with forced alignment and speaker diarization.

## Installation

WhisperX is invoked via `uvx` (UV Python package runner). No direct installation required - the module uses:
```bash
uvx whisperx <args>
```

## Usage

### Single File Transcription

```python
from src.Voice.Transcript.whisperx import transcribe_audio

# Basic transcription
result = transcribe_audio(
    file_path="audio.mp3",
    model="large-v2",
    language="pt"
)

# With speaker diarization
result = transcribe_audio(
    file_path="audio.mp3",
    model="large-v2",
    language="pt",
    diarize=True,
    hf_token="your_huggingface_token",
    min_speakers=1,
    max_speakers=4
)
```

### Batch Processing

```python
from src.Voice.Transcript.whisperx import process_audio_folder

# Process entire folder
results = process_audio_folder(
    folderpath="data/Raw/Audio",
    output_dir="data/transcripts",
    model="large-v2",
    language="pt"
)
```

## Function Reference

### `transcribe_audio`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | str | (required) | Path to input audio file |
| `model` | str | `"large-v2"` | Whisper model size |
| `align_model` | str | None | Forced alignment model (e.g., "WAV2VEC2_ASR_LARGE_LV60K_960H") |
| `batch_size` | int | None | Parallel processing batch size |
| `compute_type` | str | None | Precision type (`float16`, `int8`) |
| `language` | str | None | Language code (e.g., "pt", "en") |
| `diarize` | bool | False | Enable speaker diarization |
| `highlight_words` | bool | False | Enable word-level timestamps in subtitles |
| `min_speakers` | int | None | Minimum expected speakers |
| `max_speakers` | int | None | Maximum expected speakers |
| `hf_token` | str | None | HuggingFace token for diarization models |
| `output_dir` | str | None | Output directory for transcription files |

**Returns:** Dictionary with:
- `success` (bool): Whether transcription succeeded
- `elapsed_time` (float): Processing time in seconds
- `outputs` (dict): Paths to generated files (json, srt, txt, vtt)
- `stdout`, `stderr`, `returncode`: Process information

### `process_audio_folder`

Processes all audio files in a folder recursively.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `folderpath` | str | (required) | Directory containing audio files |
| `audio_extensions` | list | `['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4', '.avi', '.mov']` | Supported extensions |
| `output_dir` | str | None | Output directory for all transcriptions |
| `log_path` | str | None | Path to log file |
| `**kwargs` | dict | {} | Additional args passed to `transcribe_audio` |

**Returns:** Dictionary mapping file paths to transcription results.

## Output Formats

The module generates multiple output files:

| Extension | Description |
|-----------|-------------|
| `.json` | Full transcription with word-level timestamps, speakers, and metadata |
| `.srt` | SubRip subtitles with timing |
| `.vtt` | WebVTT subtitles |
| `.txt` | Plain text transcription |

## Configuration

Default settings in `conf/config.py`:
```python
WHISPER_MODEL_ID = "openai/whisper-large-v3"
TRANSCRIPTION_LANGUAGE = "portuguese"
CHUNK_LENGTH_S = 300
```

## Model Options

| Model | Size | Description |
|-------|------|-------------|
| `base` | 74M | Fastest, lowest accuracy |
| `small` | 244M | Good speed/accuracy balance |
| `medium` | 769M | Better accuracy |
| `large-v2` | 1550M | Best accuracy (default) |
| `large-v3` | 1550M | Latest Whisper large model |

## Diarization

Speaker diarization requires:
1. A HuggingFace access token
2. `diarize=True` flag
3. Optionally set `min_speakers` and `max_speakers` for better accuracy

The diarization output is included in the JSON file with speaker labels (e.g., `SPEAKER_00`, `SPEAKER_01`).