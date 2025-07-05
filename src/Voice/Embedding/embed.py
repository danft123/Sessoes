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

    def load_batch_data(self, unified_json_path, session_ids=None):
        """
        Loads a batch of metadata and audio files from a unified JSON file.
        Args:
            unified_json_path (str): Path to the unified sessions metadata JSON file.
            session_ids (list[str], optional): List of specific session_ids to load. 
                                             If None, loads all sessions.
        Returns:
            dict: A dictionary containing the loaded data for the batch, keyed by session_id.
        """
        batch_data = {}
        
        try:
            with open(unified_json_path, 'r') as f:
                all_sessions_metadata = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {unified_json_path}.")
            return batch_data
        except Exception as e:
            print(f"Error loading unified JSON file {unified_json_path}: {e}")
            return batch_data

        # Determine which sessions to process
        sessions_to_process = session_ids if session_ids else list(all_sessions_metadata.keys())
        
        print(f"\n--- Loading batch of {len(sessions_to_process)} sessions ---")
        
        for session_id in sessions_to_process:
            if session_id not in all_sessions_metadata:
                print(f"Warning: Session '{session_id}' not found in unified JSON. Skipping.")
                continue
                
            try:
                metadata = all_sessions_metadata[session_id]
                
                audio_filepath = metadata.get("filepath")
                if not audio_filepath or not os.path.exists(audio_filepath):
                    print(f"Warning: Audio file not found for session {session_id} at path {audio_filepath}. Skipping.")
                    continue

                # Load and resample audio
                waveform, sr = torchaudio.load(audio_filepath)
                if sr != self.target_sr:
                    waveform = torchaudio.functional.resample(
                                                    waveform,
                                                    orig_freq=sr,
                                                    new_freq=self.target_sr
                                                )
                    # resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
                    # waveform = resampler(waveform)

                batch_data[session_id] = {
                    "metadata": metadata,
                    "waveform": waveform
                }
                print(f"Successfully loaded data for session: {session_id}")
                
            except Exception as e:
                print(f"An unexpected error occurred while processing session {session_id}: {e}")

        return batch_data


def save_embeddings(embeddings, base_save_dir, sequence_persons=None):
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
    for i, (seg_id, emb) in enumerate(embeddings.items()):
        try:
            # session_id = seg_id[:-2] # session_id-0, session_id-1, etc. # this doesnt work because it can be session_id-112 and it would just remove 12
            session_id = seg_id.rsplit('-', 1)[0]  # Get everything before the last '-'
            session_dir = os.path.join(base_save_dir, session_id)
            os.makedirs(session_dir, exist_ok=True)

            emb_path = os.path.join(session_dir, f"{seg_id}.pt")
            torch.save(emb, emb_path)
            print(f"Saved embedding for {seg_id} to {emb_path}")
        except Exception as e:
            print(f"Error saving embedding for {seg_id}: {e}")



def compute_ecapa_tdnn_embeddings(batch_data, classifier, filter_by_person = None):
    """
    Computes ECAPA-TDNN embeddings for positive audio segments in a batch.

    Args:
        batch_data (dict): A dictionary containing the loaded audio waveforms and metadata for a batch.
            The keys are uuid session_ids and the values are dictionaries with keys 'waveform' and 'metadata' with dictionary with keys 'filepath','filename','session_id','FirstVAD','InteractiveLabeling'.
        classifier (EncoderClassifier): The pre-trained SpeechBrain model.
        filter_by_person (str, optional): If provided, only computes embeddings for segments labeled with this person's name.

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
                name = segment["name"]
                if filter_by_person and name != filter_by_person:
                    continue  # Skip segments not labeled with the specified person

                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_segment = waveform[:, start_sample:end_sample]

                if audio_segment.ndim == 1:
                    audio_segment = audio_segment.unsqueeze(0)

                with torch.no_grad():
                    embedding = classifier.encode_batch(audio_segment)
                    embedding = embedding.squeeze()

                segment_id = f"{session_id}-{name}_{i}"
                all_embeddings[segment_id] = embedding
                print(f"Computed embedding for segment: {segment_id}")

            except KeyError as e:
                print(f"Warning: Segment {i} for session {session_id} is missing key: {e}. Skipping.")
            except Exception as e:
                print(f"An error occurred while processing segment {i} for session {session_id}: {e}")

    print("Batch embedding computation finished.")
    return all_embeddings

def main(METADATA_BASE_DIR='data/labeling', SESSIONS_METADATA_FILENAME='sessions_metadata_20250705_120500.json', BATCH_SIZE=1):
    # --- Configuration ---
    UNIFIED_JSON_PATH = os.path.join(METADATA_BASE_DIR, SESSIONS_METADATA_FILENAME)
    EMBEDDINGS_SAVE_DIR = METADATA_BASE_DIR + '/embeddings'
    # --- Pre-computation Setup ---
    # Check if the unified JSON file exists
    if not os.path.exists(UNIFIED_JSON_PATH):
        print(f"Error: Unified metadata file not found at {UNIFIED_JSON_PATH}. Halting execution.")
        exit()

    # Load the unified JSON to get all session IDs
    try:
        with open(UNIFIED_JSON_PATH, 'r') as f:
            all_sessions_metadata = json.load(f)
        all_session_ids = list(all_sessions_metadata.keys())
    except Exception as e:
        print(f"Error loading unified JSON file: {e}")
        exit()

    if not all_session_ids:
        print(f"Error: No sessions found in {UNIFIED_JSON_PATH}. Halting execution.")
        exit()

    print(f"Found {len(all_session_ids)} total sessions to process.")

    # # Get names of persons in the metadata
    # sequence_persons = {}
    # for session_id, metadata in all_sessions_metadata.items():
    #     sequence_persons[session_id] = []
    #     if "InteractiveLabeling" in metadata:
    #         positive_segments = metadata["InteractiveLabeling"].get("positive_segments", [])
    #         for segment in positive_segments:
    #             person_name = segment.get("name", None)
    #             if person_name:
    #                 sequence_persons[session_id].append(person_name)
    # # get unique persons
    # unique_persons = set()
    # for persons in sequence_persons.values():
    #     unique_persons.update(persons)
    # print(f"Unique persons found in metadata: {', '.join(unique_persons)}")

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

    # Process sessions in batches
    for i in range(0, len(all_session_ids), BATCH_SIZE):
        batch_session_ids = all_session_ids[i:i + BATCH_SIZE]
        print(f"\n--- Processing Batch {i//BATCH_SIZE + 1}/{(len(all_session_ids) + BATCH_SIZE - 1)//BATCH_SIZE} ---")

        # 1. Load the data for the current batch
        loaded_batch_data = loader.load_batch_data(UNIFIED_JSON_PATH, batch_session_ids)

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


if __name__ == '__main__':
    main()