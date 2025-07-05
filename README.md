# Config Files (./conf/config.py)
```python
# --- Directory Paths ---
INPUT_AUDIO_DIR = "data/Raw/ProcessedMp3s"
OUTPUT_TRANSCRIPT_DIR = "data/labeling"
TEMP_DIR = "data/temp" # For masked files and other temporary assets
```

# To label the audio files in the folder .data/Raw/ProcessedMp3s
```python
from src.Voice.Embedding import LabelFolder
LabelFolder()  # Call the function to start the labeling process
```

Files exist now in ./data/transcripts in the format uuid_metadata.json, where uuid is the unique identifier for the audio file and metadata contains the labels and other information.