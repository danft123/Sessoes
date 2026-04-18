# Sessões

Pipeline to process audio recordings from game/brainstorm sessions into structured information.

## Overview

This project transforms raw OBS audio recordings into searchable, structured insights using:
- Voice activity detection & speaker diarization
- Interactive speaker labeling
- Speaker embeddings (ECAPA-TDNN)
- LLM-powered analysis

## Pipeline

```
Raw Video/Audio → WhisperX Transcription → Interactive Labeling → LLM Processing
```

### 1. Video/Audio Loading
Loads video files from OBS recordings and extracts audio track.

**Module:** `src/VideoLoader/loader.py`

### 2. Transcription
Transcribes audio using WhisperX with word-level timestamps and speaker diarization.

**Module:** `src/Voice/Transcript/whisperx.py`

**Output formats:** JSON, SRT, VTT, TXT

```python
from src.Voice.Transcript.whisperx import process_audio_folder

process_audio_folder("data/Raw/Audio", output_dir="data/transcripts")
```

### 3. Interactive Labeling
Human-in-the-loop labeling to identify speakers and filter noise.

**Module:** `src/Voice/Embedding/label_audio.py`

Flow:
1. Voice segments detected via WebRTC VAD
2. Each segment played interactively
3. User confirms positive (speaker name) or negative (reason)
4. Negative segments filtered through VAD

```python
from src.Voice.Embedding import LabelFolder

LabelFolder()  # Labels all MP3s in config.INPUT_AUDIO_DIR
```

### 4. Speaker Embeddings
Computes ECAPA-TDNN embeddings for labeled voice segments.

**Module:** `src/Voice/Embedding/embed.py`

Output: `.pt` embedding files per session

## LLM Agents

Processes transcripts through Claude (Sonnet 4) for analysis.

**Module:** `src/LLM/utils.py`

### Summary Generator
Creates concise summaries with source timestamps.

```python
from src.LLM.utils import transcription_sumary

summary = transcription_sumary(transcription_text)
```

### Progress Report Extractor
Extracts progress reports, key points, and actions.

```python
from src.LLM.utils import transcription_progress_report

report = transcription_progress_report(transcription_text)
```

### Blob of Idea Capture
Extracts "blobs" - self-contained ideas the user wants to save.

**Tags:** `[progress_report]`, `[action_item]`, `[key_decision]`, `[new_idea]`

```python
from src.LLM.utils import transcription_capture_tags

blobs = transcription_capture_tags(transcription_text)
```

## Configuration

**Directory paths** (`conf/config.py`):

| Variable | Default |
|----------|---------|
| `INPUT_AUDIO_DIR` | `data/Raw/ProcessedMp3s` |
| `OUTPUT_TRANSCRIPT_DIR` | `data/transcripts` |
| `OUTPUT_LABEL_DIR` | `data/labeling` |
| `TEMP_DIR` | `data/temp` |

**Key parameters:**
- `SAMPLE_RATE = 16000`
- `WHISPER_MODEL_ID = openai/whisper-large-v3`
- `TRANSCRIPTION_LANGUAGE = portuguese`