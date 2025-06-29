import torch
import gc
import numpy as np
from tqdm import tqdm

def chunk_audio(audio_array, chunk_size, sampling_rate, overlap=0):
    """
    Chunks an audio array into smaller pieces with optional overlap.

    Args:
        audio_array (np.ndarray): The input audio array.
        chunk_size (int): The size of each chunk in seconds.
        sampling_rate (int): The sampling rate of the audio.
        overlap (int): The overlap between chunks in seconds.

    Yields:
        np.ndarray: A chunk of the audio array.
    """
    total_samples = audio_array.shape[-1]
    chunk_samples = int(chunk_size * sampling_rate)
    overlap_samples = int(overlap * sampling_rate)
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        yield audio_array[start:end]
        if end == total_samples:
            break
        start = end - overlap_samples

def transcribe_with_progress(audio_dict, pipe, generate_kwargs, device, chunk_length_s=30, overlap_s=2, USE_MODEL_CHUNKER = False):
    """
    Transcribes an audio file chunk by chunk with a progress bar and memory management.
    This version properly handles timestamps by adjusting them for each chunk's position.

    Args:
        audio_dict (dict): A dictionary containing the audio array and sampling rate.
        pipe (pipeline): The Hugging Face ASR pipeline.
        generate_kwargs (dict): Arguments for the model's generate function.
        device (str): The device to use for processing ('cuda' or 'cpu').
        chunk_length_s (int): The length of each audio chunk for processing.
        overlap_s (int): The overlap between chunks in seconds.
        torch_dtype (torch.dtype): The torch data type to use for the audio array.

    Returns:
        dict: Dictionary containing 'text' and 'chunks' with timestamps.
    """
    audio_array = audio_dict["array"]
    sampling_rate = audio_dict["sampling_rate"]
    # Create chunks from the audio array
    chunks = list(chunk_audio(audio_array, chunk_length_s, sampling_rate, overlap=overlap_s))
    
    all_chunks = []
    full_text = []
    chunk_start_time = 0.0
    overlap_samples = int(overlap_s * sampling_rate)
    
    # Process each chunk with a progress bar
    for i, chunk in enumerate(tqdm(chunks, desc="Transcribing", unit="chunk")):
        try:
            # Create a proper audio dictionary for the pipeline
            chunk_dict = {
                "array": chunk,
                "sampling_rate": sampling_rate
            }
            
            # Use torch.no_grad() to save memory during inference
            with torch.no_grad():
                # Use the pipeline with return_timestamps=True directly
                result = pipe(
                    chunk_dict,
                    return_timestamps=True,
                    generate_kwargs=generate_kwargs,
                )
                
                # Process the result
                if isinstance(result, dict):
                    text = result.get("text", "")
                    chunks_with_timestamps = result.get("chunks", [])
                    
                    # Adjust timestamps to account for chunk position in full audio
                    for chunk_info in chunks_with_timestamps:
                        if "timestamp" in chunk_info and chunk_info["timestamp"]:
                            start_time, end_time = chunk_info["timestamp"]
                            if start_time is not None:
                                start_time += chunk_start_time
                            if end_time is not None:
                                end_time += chunk_start_time
                            
                            adjusted_chunk = {
                                "text": chunk_info["text"],
                                "timestamp": [start_time, end_time]
                            }
                            all_chunks.append(adjusted_chunk)
                    
                    if text.strip():
                        full_text.append(text.strip())
                        
                else:
                    # Fallback for simpler result format
                    text = str(result).strip()
                    if text:
                        full_text.append(text)
                        all_chunks.append({
                            "text": text,
                            "timestamp": [chunk_start_time, chunk_start_time + len(chunk) / sampling_rate]
                        })
                        
        except Exception as e:
            print(f"Error transcribing chunk {i}: {e}")
            continue
        finally:
            # Update chunk start time for next iteration
            if i < len(chunks) - 1:  # Not the last chunk
                chunk_start_time += (len(chunk) - overlap_samples) / sampling_rate
            
            # Force garbage collection and clear CUDA cache after each chunk
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    return {
        "text": " ".join(full_text),
        "chunks": all_chunks
    }

def remap_timestamps(transcription_result, original_timestamps):
    """
    Remaps timestamps from a concatenated file back to the original timeline.
    Safely handles None values in timestamps.
    """
    if not transcription_result or "chunks" not in transcription_result:
        return {"remapped_chunks": []}

    segment_durations = [end - start for start, end in original_timestamps]
    cumulative_durations = [0] + list(np.cumsum(segment_durations))

    remapped_chunks = []
    for chunk in transcription_result.get("chunks", []):
        chunk_start, chunk_end = chunk.get("timestamp", (None, None))
        
        if chunk_start is None:
            continue

        # Find which original segment this chunk belongs to
        seg_idx = -1
        for i, cum_dur in enumerate(cumulative_durations[:-1]):
            if chunk_start >= cum_dur:
                seg_idx = i

        if seg_idx != -1:
            original_start, _ = original_timestamps[seg_idx]
            offset = cumulative_durations[seg_idx]
            
            real_start = original_start + (chunk_start - offset)
            real_end = original_start + (chunk_end - offset) if chunk_end is not None else None
            
            remapped_chunks.append({
                "text": chunk["text"],
                "real_timestamp": (real_start, real_end)
            })

    return {"remapped_chunks": remapped_chunks}