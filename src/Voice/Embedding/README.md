# Voice.Embedding

Speaker embedding computation and interactive labeling for voice segments.

## Overview

This module provides two main functionalities:
1. **Interactive Labeling** - Human-in-the-loop labeling to identify speakers and filter noise
2. **Speaker Embeddings** - Compute ECAPA-TDNN embeddings for labeled voice segments

## Installation

Required dependencies:
```bash
pip install pygame webrtcvad speechbrain torchaudio librosa soundfile
```

## Interactive Labeling

### Usage

```python
from src.Voice.Embedding import LabelFolder

# Labels all MP3s in config.INPUT_AUDIO_DIR
LabelFolder()
```

Or directly:
```python
from src.Voice.Embedding.label_audio import main

main()
```

### Flow

1. **Voice Detection** - Segments are detected via WebRTC VAD
2. **Interactive Review** - Each segment is played automatically
3. **User Decision** - Confirm as positive (speaker name) or negative (reason)
4. **VAD Filtering** - Negative segments are filtered through additional VAD checks

### Controls

| Input | Action |
|-------|--------|
| `y` | Approve segment, enter speaker name |
| `n` | Skip segment (marked as negative) |
| Enter | Replay current segment |
| `q` | Quit labeling |

### Output

Labeled segments are saved to `data/labeling/sessions_metadata_<timestamp>.json`:

```json
{
  "session_uuid": {
    "filepath": "/path/to/audio.mp3",
    "filename": "audio.mp3",
    "session_id": "uuid",
    "FirstVAD": [[start, end], ...],
    "InteractiveLabeling": {
      "positive_segments": [
        {"start": 0.5, "end": 2.3, "name": "Daniel", "duration": 1.8}
      ],
      "negative_segments": [
        {"start": 5.0, "end": 5.5, "reason": "noise", "duration": 0.5}
      ]
    }
  }
}
```

## Speaker Embeddings

### Usage

```python
from src.Voice.Embedding import EmbedLabeledVoiceSegments

# Or run directly:
# python -m src.Voice.Embedding.embed
```

### Configuration

Update the main function parameters:
```python
main(
    METADATA_BASE_DIR='data/labeling',
    SESSIONS_METADATA_FILENAME='sessions_metadata_20250705_120500.json',
    BATCH_SIZE=1
)
```

### Output

Embeddings are saved as `.pt` files in `data/labeling/embeddings/<session_id>/`:
```
data/labeling/embeddings/<session_id>/<segment_id>.pt
```

Each file contains a PyTorch tensor of shape `(1, 192)` (ECAPA-TDNN embedding dimension).

## Classes

### AudioMetadataLoader

Loads audio files and metadata for embedding computation.

```python
loader = AudioMetadataLoader(target_sr=16000)
batch_data = loader.load_batch_data(
    unified_json_path="data/labeling/sessions_metadata.json",
    session_ids=None  # Optional: filter specific sessions
)
```

### Functions

| Function | Description |
|----------|-------------|
| `label_voice_segments_interactively` | Interactive labeling of voice segments |
| `check_speech_in_segment` | VAD-based speech detection |
| `compute_ecapa_tdnn_embeddings` | Compute embeddings for positive segments |
| `save_embeddings` | Save embeddings as .pt files |

## Configuration

From `conf/config.py`:

```python
SAMPLE_RATE = 16000
VAD_AGGRESSIVENESS = 1
VAD_FRAME_DURATION_MS = 30
VAD_VOICE_THRESHOLD = 0.6
VAD_MIN_SPEECH_DURATION_S = 1.0
```

## Model

The module uses **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in Time Delay Neural Network) via SpeechBrain:

- Source: `speechbrain/spkrec-ecapa-voxceleb`
- Embedding dimension: 192
- Sample rate: 16000 Hz

## Combined Usage

```python
from src.Voice.Embedding import LabelAndEmbed

# Run entire pipeline: labeling + embeddings
LabelAndEmbed()
```