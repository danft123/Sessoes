# Voice

Voice processing module for audio transcription, speaker labeling, and embedding computation.

## Overview

The Voice module is the core component for processing audio recordings. It provides:

- **Transcription** - Convert audio to text with WhisperX
- **Speaker Labeling** - Interactive human-in-the-loop labeling
- **Speaker Embeddings** - ECAPA-TDNN embeddings for speaker identification

## Structure

```
Voice/
├── Transcript/       # Audio transcription with WhisperX
│   └── whisperx.py
└── Embedding/       # Speaker labeling and embeddings
    ├── label_audio.py
    └── embed.py
```

## Quick Start

### Transcription

```python
from src.Voice.Transcript import transcribe_audio

result = transcribe_audio("audio.mp3", language="pt")
```

### Labeling

```python
from src.Voice.Embedding import LabelFolder

LabelFolder()  # Interactively label speakers
```

### Embeddings

```python
from src.Voice.Embedding import EmbedLabeledVoiceSegments

EmbedLabeledVoiceSegments()
```

### Complete Pipeline

```python
from src.Voice.Embedding import LabelAndEmbed

LabelAndEmbed()  # Label + compute embeddings
```

## Dependencies

- `whisperx` - Transcription (via uvx)
- `speechbrain` - ECAPA-TDNN embeddings
- `pygame` - Audio playback for labeling
- `webrtcvad` - Voice activity detection
- `librosa`, `soundfile` - Audio file handling

See `conf/config.py` for configuration options.