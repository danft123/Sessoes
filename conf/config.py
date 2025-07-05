import torch

# --- Directory Paths ---
INPUT_AUDIO_DIR = "data/Raw/ProcessedMp3s"
OUTPUT_TRANSCRIPT_DIR = "data/transcripts"
OUTPUT_LABEL_DIR = "data/labeling"
TEMP_DIR = "data/temp" # For masked files and other temporary assets

# --- Voice Masking Parameters (voice_mask.py) ---
PERSON = "Daniel"  # The person to create a mask for, must match with the labeled audio metadata.
# VAD (Voice Activity Detection) Parameters
VAD_AGGRESSIVENESS = 1                 # 0-3: How aggressive VAD is
VAD_FRAME_DURATION_MS = 30             # 10, 20, or 30
VAD_VOICE_THRESHOLD = 0.6              # Confidence threshold for voice activity
VAD_MIN_SPEECH_DURATION_S = 1.0        # Minimum duration to be considered speech
ENERGY_FILTERING_ENABLED = True        # Whether to filter by audio energy
ENERGY_PERCENTILE_THRESHOLD = 30       # Energy percentile to filter quiet segments
SPECTRAL_CLUSTERING_ENABLED = False     # Use spectral clustering if no personal model
EMBEDDING_CLUSTERING_ENABLED = True
FADE_DURATION_S = 0.1                  # Fade in/out for combined audio to avoid clicks

# --- Transcription Parameters (transcribe.py) ---
# Available models: "openai/whisper-large-v3", "openai/whisper-medium", "openai/whisper-small"
WHISPER_MODEL_ID = "openai/whisper-large-v3"
TRANSCRIPTION_LANGUAGE = "portuguese"
TRANSCRIPTION_TASK = "transcribe"
CHUNK_LENGTH_S = 300                    # Length of audio chunks for transcription
CHUNK_OVERLAP_S = 5                      # Overlap between chunks to ensure context

# --- Device and Data Type ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# --- Audio Parameters ---
SAMPLE_RATE = 16000