import os
import json
import glob
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

class AudioMetadataLoader:
    """
    A class to load a batch of audio files and their corresponding metadata.
    """
    def __init__(self, target_sr=16000):
        """
        Initializes the DataLoader.

        Args:
            target_sr (int): The target sampling rate to resample the audio to.
        """
        self.target_sr = target_sr

    def load_batch_data(self, metadata_files):
        """
        Loads a batch of metadata and audio files from a given list of file paths.

        Args:
            metadata_files (list[str]): A list of paths to metadata files to load.

        Returns:
            dict: A dictionary containing the loaded data for the batch, keyed by session_id.
        """
        batch_data = {}
        print(f"\n--- Loading batch of {len(metadata_files)} files ---")
        for meta_file in metadata_files:
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)

                session_id = metadata.get("session_id")
                if not session_id:
                    print(f"Warning: 'session_id' not found in {meta_file}. Skipping.")
                    continue

                audio_filepath = metadata.get("filepath")
                if not audio_filepath or not os.path.exists(audio_filepath):
                    print(f"Warning: Audio file not found for {meta_file} at path {audio_filepath}. Skipping.")
                    continue

                # Load and resample audio
                waveform, sr = torchaudio.load(audio_filepath)
                if sr != self.target_sr:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                    waveform = resampler(waveform)

                batch_data[session_id] = {
                    "metadata": metadata,
                    "waveform": waveform
                }
                print(f"Successfully loaded data for session: {session_id}")

            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {meta_file}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred while processing {meta_file}: {e}")
        return batch_data

def compute_ecapa_tdnn_embeddings(batch_data, classifier):
    """
    Computes ECAPA-TDNN embeddings for positive audio segments in a batch.

    Args:
        batch_data (dict): A dictionary containing the loaded audio waveforms and metadata for a batch.
        classifier (EncoderClassifier): The pre-trained SpeechBrain model.

    Returns:
        dict: A dictionary where keys are segment identifiers (session_id-{i}) and
              values are the computed embeddings as PyTorch tensors.
    """
    all_embeddings = {}
    print("Starting embedding computation for the batch...")

    for session_id, data in batch_data.items():
        waveform = data["waveform"]
        metadata = data["metadata"]
        sr = 16000  # The model and our data are at 16000 Hz

        positive_segments = metadata.get("InteractiveLabeling", {}).get("positive_segments", [])

        if not positive_segments:
            print(f"No positive segments found for session: {session_id}")
            continue

        for i, segment in enumerate(positive_segments):
            try:
                start_time = segment["start"]
                end_time = segment["end"]

                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_segment = waveform[:, start_sample:end_sample]

                if audio_segment.ndim == 1:
                    audio_segment = audio_segment.unsqueeze(0)

                with torch.no_grad():
                    embedding = classifier.encode_batch(audio_segment)
                    embedding = embedding.squeeze()

                segment_id = f"{session_id}-{i}"
                all_embeddings[segment_id] = embedding
                print(f"Computed embedding for segment: {segment_id}")

            except KeyError as e:
                print(f"Warning: Segment {i} for session {session_id} is missing key: {e}. Skipping.")
            except Exception as e:
                print(f"An error occurred while processing segment {i} for session {session_id}: {e}")

    print("Batch embedding computation finished.")
    return all_embeddings

def save_embeddings(embeddings, base_save_dir):
    """
    Saves computed embeddings to .pt files.

    Args:
        embeddings (dict): A dictionary of segment_id -> embedding_tensor.
        base_save_dir (str): The base directory to save the embedding files.
    """
    if not embeddings:
        print("No embeddings to save for this batch.")
        return

    print(f"--- Saving {len(embeddings)} embeddings ---")
    for seg_id, emb in embeddings.items():
        try:
            # Assuming seg_id is in the format 'session_id-...'
            session_id = seg_id.split('-')[0]
            session_dir = os.path.join(base_save_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)

            emb_path = os.path.join(session_dir, f"{seg_id}.pt")
            torch.save(emb, emb_path)
            print(f"Saved embedding for {seg_id} to {emb_path}")
        except Exception as e:
            print(f"Error saving embedding for {seg_id}: {e}")


if __name__ == '__main__':
    # --- Configuration ---
    METADATA_BASE_DIR = 'data/transcripts'
    EMBEDDINGS_SAVE_DIR = 'data/embeddings'
    BATCH_SIZE = 2 # Process 5 files at a time to manage memory

    # --- Pre-computation Setup ---
    # Find all metadata files to be processed
    all_metadata_files = glob.glob(os.path.join(METADATA_BASE_DIR, '*_metadata.json'))

    if not all_metadata_files:
        print(f"Error: No metadata files found in {METADATA_BASE_DIR}. Halting execution.")
        exit()

    print(f"Found {len(all_metadata_files)} total metadata files to process.")

    # Load the model once to avoid reloading it for every batch
    print("\nInitializing ECAPA-TDNN model...")
    try:
        model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Fatal Error: Could not load the SpeechBrain model. Please check your internet connection and dependencies. Error: {e}")
        exit()

    # --- Main Execution Loop ---
    loader = AudioMetadataLoader(target_sr=16000)
    total_embeddings_processed = 0

    # Process files in batches
    for i in range(0, len(all_metadata_files), BATCH_SIZE):
        batch_files = all_metadata_files[i:i + BATCH_SIZE]
        print(f"\n--- Processing Batch {i//BATCH_SIZE + 1}/{(len(all_metadata_files) + BATCH_SIZE - 1)//BATCH_SIZE} ---")

        # 1. Load the data for the current batch
        loaded_batch_data = loader.load_batch_data(batch_files)

        if not loaded_batch_data:
            print("No data was loaded in this batch. Skipping to next.")
            continue

        # 2. Compute embeddings for the batch
        embeddings_batch = compute_ecapa_tdnn_embeddings(loaded_batch_data, model)

        # 3. Save the computed embeddings
        save_embeddings(embeddings_batch, EMBEDDINGS_SAVE_DIR)
        
        total_embeddings_processed += len(embeddings_batch)

    print(f"\n--- All Batches Processed ---")
    print(f"Total embeddings computed and saved: {total_embeddings_processed}")

