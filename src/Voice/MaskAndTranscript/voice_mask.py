import librosa
import numpy as np
import os
import soundfile as sf
from sklearn.cluster import KMeans
import webrtcvad
from typing import List, Tuple, Optional

# Import settings from the central config file
from conf.config import (
    SAMPLE_RATE, VAD_AGGRESSIVENESS, VAD_FRAME_DURATION_MS, VAD_VOICE_THRESHOLD,
    VAD_MIN_SPEECH_DURATION_S, ENERGY_FILTERING_ENABLED, ENERGY_PERCENTILE_THRESHOLD,
    SPECTRAL_CLUSTERING_ENABLED, FADE_DURATION_S, PERSON
)

def create_voice_mask(
    audio_file: str,
    sample_rate: int = SAMPLE_RATE,
    frame_duration: int = VAD_FRAME_DURATION_MS,
    voice_threshold: float = VAD_VOICE_THRESHOLD,
    min_speech_duration: float = VAD_MIN_SPEECH_DURATION_S,
    use_energy_filtering: bool = ENERGY_FILTERING_ENABLED,
    energy_percentile: float = ENERGY_PERCENTILE_THRESHOLD,
    use_spectral_clustering: bool = SPECTRAL_CLUSTERING_ENABLED
) -> List[Tuple[float, float]]:
    """
    Creates a mask of timestamps where a user is likely speaking.
    """
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_length = int(sample_rate * frame_duration / 1000)
    
    voice_segments = []
    current_start = None
    
    for i in range(0, len(audio) - frame_length, frame_length):
        frame_bytes = (audio[i:i + frame_length] * 32767).astype(np.int16).tobytes()
        is_speech = vad.is_speech(frame_bytes, sample_rate)
        timestamp = i / sample_rate
        
        if is_speech and current_start is None:
            current_start = timestamp
        elif not is_speech and current_start is not None:
            if timestamp - current_start >= min_speech_duration:
                voice_segments.append((current_start, timestamp))
            current_start = None
            
    if current_start is not None and (len(audio) / sample_rate) - current_start >= min_speech_duration:
        voice_segments.append((current_start, len(audio) / sample_rate))
        
    if use_energy_filtering:
        voice_segments = _filter_by_energy(audio, voice_segments, sr, energy_percentile)
        
    if use_spectral_clustering and len(voice_segments) > 1:
        voice_segments = _filter_by_spectral_clustering(audio, voice_segments, sr)
        
    return voice_segments


def _filter_by_energy(
    audio: np.ndarray, 
    segments: List[Tuple[float, float]], 
    sr: int,
    percentile: float
) -> List[Tuple[float, float]]:
    """Filter segments by audio energy levels."""
    if not segments:
        return segments
    
    # Calculate energy for each segment
    segment_energies = []
    for start, end in segments:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        segment_audio = audio[start_idx:end_idx]
        energy = np.sum(segment_audio ** 2) / len(segment_audio)
        segment_energies.append(energy)
    
    # Filter by energy threshold
    energy_threshold = np.percentile(segment_energies, percentile)
    filtered_segments = [
        seg for seg, energy in zip(segments, segment_energies)
        if energy >= energy_threshold
    ]
    
    return filtered_segments


def _filter_by_spectral_clustering(
    audio: np.ndarray,
    segments: List[Tuple[float, float]],
    sr: int
) -> List[Tuple[float, float]]:
    """Use spectral features to identify consistent voice characteristics."""
    if len(segments) < 2:
        return segments
    
    # Extract MFCC features for each segment
    features = []
    for start, end in segments:
        start_idx = int(start * sr)
        end_idx = int(end * sr)
        segment_audio = audio[start_idx:end_idx]
        
        if len(segment_audio) > sr * 0.5:  # At least 0.5 seconds
            mfccs = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=13)
            # Use mean and std of MFCCs as features
            feature_vector = np.concatenate([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1)
            ])
            features.append(feature_vector)
        else:
            features.append(None)
    
    # Filter out None features
    valid_features = [(i, f) for i, f in enumerate(features) if f is not None]
    if len(valid_features) < 2:
        return segments
    
    indices, feature_vectors = zip(*valid_features)
    feature_matrix = np.array(feature_vectors)
    
    # Cluster into 2 groups: your voice vs others
    kmeans = KMeans(n_clusters=min(2, len(feature_vectors)), random_state=42)
    clusters = kmeans.fit_predict(feature_matrix)
    
    # Assume the larger cluster is your voice (you speak more)
    cluster_counts = np.bincount(clusters)
    primary_cluster = np.argmax(cluster_counts)
    
    # Return segments from the primary cluster
    filtered_segments = [
        segments[idx] for i, idx in enumerate(indices)
        if clusters[i] == primary_cluster
    ]
    
    return filtered_segments

def export_audio_segments(
    audio_file: str,
    timestamps: List[Tuple[float, float]],
    output_dir: str,
    fade_duration: float = FADE_DURATION_S,
    sample_rate: int = SAMPLE_RATE
):
    """
    Exports individual audio segments from an input file based on timestamps.
    Each segment is saved as a separate file with fade in/out applied.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    fade_samples = int(fade_duration * sr)

    exported_files = []

    for idx, (start, end) in enumerate(timestamps):
        start_idx, end_idx = int(start * sr), int(end * sr)
        segment = audio[start_idx:end_idx].copy()

        if len(segment) > 2 * fade_samples:
            segment[:fade_samples] *= np.linspace(0, 1, fade_samples)
            segment[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        out_filename = f"{os.path.splitext(os.path.basename(audio_file))[0]}_{start:.2f}-{end:.2f}.wav"
        out_path = os.path.join(output_dir, out_filename)
        sf.write(out_path, segment, sr)
        exported_files.append(out_path)

    if exported_files:
        print(f"{len(exported_files)} segments saved to {output_dir}")
    else:
        print("No segments were exported.")

    return exported_files

def create_combined_audio(
    audio_file: str,
    timestamps: List[Tuple[float, float]],
    output_file: str,
    fade_duration: float = FADE_DURATION_S,
    sample_rate: int = SAMPLE_RATE
):
    """
    Creates a single audio file from concatenated speech segments.
    """
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    combined_audio = []
    fade_samples = int(fade_duration * sr)

    for start, end in timestamps:
        start_idx, end_idx = int(start * sr), int(end * sr)
        segment = audio[start_idx:end_idx].copy()
        
        if len(segment) > 2 * fade_samples:
            segment[:fade_samples] *= np.linspace(0, 1, fade_samples)
            segment[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        combined_audio.append(segment)
        
    if combined_audio:
        final_audio = np.concatenate(combined_audio)
        sf.write(output_file, final_audio, sr)
        print(f"Combined audio saved to {output_file}")
        return output_file
    else:
        print("No segments to combine.")
        return None