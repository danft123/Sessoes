import librosa
import numpy as np
import os
import soundfile as sf
from sklearn.cluster import KMeans
import webrtcvad
from typing import List, Tuple, Optional
import torch
import torchaudio
from speechbrain.inference import EncoderClassifier

def compute_ecapa_tdnn_embedding(audio_segment: torch.Tensor, model: EncoderClassifier) -> torch.Tensor:
    if audio_segment.ndim == 1:
        audio_segment = audio_segment.unsqueeze(0)

    with torch.no_grad():
        embedding = model.encode_batch(audio_segment)
        embedding = embedding.squeeze()
    return embedding

# Import settings from the central config file
from conf.config import (
    SAMPLE_RATE, VAD_AGGRESSIVENESS, VAD_FRAME_DURATION_MS, VAD_VOICE_THRESHOLD,
    VAD_MIN_SPEECH_DURATION_S, ENERGY_FILTERING_ENABLED, ENERGY_PERCENTILE_THRESHOLD,
    SPECTRAL_CLUSTERING_ENABLED, FADE_DURATION_S, EMBEDDING_CLUSTERING_ENABLED, PERSON, OUTPUT_LABEL_DIR
)

def create_voice_mask(
    audio_file: str,
    sample_rate: int = SAMPLE_RATE,
    frame_duration: int = VAD_FRAME_DURATION_MS,
    voice_threshold: float = VAD_VOICE_THRESHOLD,
    min_speech_duration: float = VAD_MIN_SPEECH_DURATION_S,
    use_energy_filtering: bool = ENERGY_FILTERING_ENABLED,
    energy_percentile: float = ENERGY_PERCENTILE_THRESHOLD,
    use_spectral_clustering: bool = SPECTRAL_CLUSTERING_ENABLED,
    use_embedding_clustering: bool = EMBEDDING_CLUSTERING_ENABLED,
) -> List[Tuple[float, float]]:
    """
    Creates a mask of timestamps where a user is likely speaking.
    """
    # audio, sr = librosa.load(audio_file, sr=sample_rate)
    _audio, sr = torchaudio.load(audio_file, channels_first = False)
    audio = _audio.mean(dim=1) if _audio.ndim > 1 else _audio
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_length = int(sample_rate * frame_duration / 1000)
    
    voice_segments = []
    current_start = None
    
    for i in range(0, len(audio) - frame_length, frame_length):
        frame_bytes = (audio[i:i + frame_length] * 32767).numpy().astype(np.int16).tobytes()
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
        voice_segments = _filter_by_energy(audio.numpy(), voice_segments, sr, energy_percentile)
        
    if use_spectral_clustering and len(voice_segments) > 1:
        voice_segments = _filter_by_spectral_clustering(audio.numpy(), voice_segments, sr)
    
    if use_embedding_clustering and len(voice_segments) > 1:
        model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        # load embs, which is every .pt object in (use glob since they are inside folders) OUTPUT_LABEL_DIR+'/embeddings with 'PERSON' in the name
        embs = []
        for root, dirs, files in os.walk(OUTPUT_LABEL_DIR):
            for file in files:
                if PERSON in file and file.endswith('.pt'):
                    emb_path = os.path.join(root, file)
                    _emb = torch.load(emb_path)
                    # change to mono, it is of shape (C,T)
                    emb = _emb.mean(dim=0) if _emb.ndim > 1 else _emb
                    embs.append(emb)
        voice_segments = _filter_by_embedding_clustering(audio, voice_segments, model, embs)
        
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

import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import torch

def _filter_by_embedding_clustering(
    audio: torch.Tensor,
    segments: List[Tuple[float, float]],
    model: EncoderClassifier,
    embs: List[torch.Tensor],
    similarity_threshold: float = 0.7,
    min_samples: int = 2
) -> List[Tuple[float, float]]:
    """
    Use ECAPA-TDNN embeddings to filter segments based on speaker characteristics.
    
    Args:
        audio: Audio signal
        segments: List of (start, end) time segments
        model: ECAPA-TDNN model for embedding extraction
        embs: List of reference embeddings for the target speaker
        similarity_threshold: Minimum cosine similarity for speaker verification
        min_samples: Minimum samples for DBSCAN clustering
        
    Returns:
        Filtered segments containing only the target speaker
    """
    if len(segments) < 2:
        return segments
    
    # Extract embeddings from candidate segments
    embeddings = []
    valid_segments = []
    
    for i, (start, end) in enumerate(segments):
        start_idx = int(start * SAMPLE_RATE)
        end_idx = int(end * SAMPLE_RATE)
        # audio is of shape (T, C), we cut from start_idx to end_idx
        segment_audio = audio[start_idx:end_idx]
        
        if len(segment_audio) > SAMPLE_RATE * 0.5:  # At least 0.5 seconds
            embedding = compute_ecapa_tdnn_embedding(segment_audio, model)
            embeddings.append(embedding)
            valid_segments.append((start, end))
    
    if not embeddings or not embs:
        return segments
    
    # Convert to numpy arrays for processing
    segment_embeddings = embeddings
    reference_embeddings = embs
    
    # Method 1: Direct similarity comparison with reference embeddings
    filtered_segments_similarity = []
    for i, seg_emb in enumerate(segment_embeddings):
        # Calculate cosine similarity with all reference embeddings
        similarities = cosine_similarity([seg_emb], reference_embeddings)[0]
        max_similarity = np.max(similarities)
        
        if max_similarity >= similarity_threshold:
            filtered_segments_similarity.append(valid_segments[i])
    
    # Method 2: Enhanced clustering approach with reference embeddings
    # Combine reference embeddings with segment embeddings for clustering
    all_embeddings = np.vstack([reference_embeddings, segment_embeddings])
    
    # Use DBSCAN for clustering (handles noise and varying cluster sizes)
    clustering = DBSCAN(
        eps=1 - similarity_threshold,  # Convert similarity to distance
        min_samples=min_samples,
        metric='cosine'
    ).fit(all_embeddings)
    
    # Find which cluster(s) contain the reference embeddings
    ref_labels = clustering.labels_[:len(reference_embeddings)]
    target_clusters = set(ref_labels[ref_labels >= 0])  # Exclude noise (-1)
    
    # Filter segments based on clustering results
    filtered_segments_clustering = []
    segment_labels = clustering.labels_[len(reference_embeddings):]
    
    for i, label in enumerate(segment_labels):
        if label in target_clusters and label >= 0:  # Not noise
            filtered_segments_clustering.append(valid_segments[i])
    
    # Method 3: Ensemble approach - combine both methods
    # Take intersection of both methods for high confidence
    similarity_set = set(filtered_segments_similarity)
    clustering_set = set(filtered_segments_clustering)
    
    # Use similarity method as primary, clustering as validation
    # If clustering gives significantly fewer results, trust similarity more
    if len(clustering_set) < len(similarity_set) * 0.5:
        final_segments = filtered_segments_similarity
    else:
        # Take intersection for high confidence
        final_segments = list(similarity_set.intersection(clustering_set))
        
        # If intersection is too restrictive, fall back to similarity method
        if len(final_segments) < len(similarity_set) * 0.3:
            final_segments = filtered_segments_similarity
    
    return final_segments


def compute_speaker_embedding_statistics(embs: List[np.ndarray]) -> dict:
    """
    Compute statistics for reference embeddings to help with threshold tuning.
    
    Args:
        embs: List of reference embeddings
        
    Returns:
        Dictionary with embedding statistics
    """
    if not embs:
        return {}
    
    embeddings = np.array(embs)
    
    # Compute pairwise similarities within reference embeddings
    similarities = cosine_similarity(embeddings)
    
    # Remove diagonal (self-similarity)
    mask = np.ones(similarities.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    similarities_flat = similarities[mask]
    
    stats = {
        'mean_intra_similarity': np.mean(similarities_flat),
        'std_intra_similarity': np.std(similarities_flat),
        'min_intra_similarity': np.min(similarities_flat),
        'max_intra_similarity': np.max(similarities_flat),
        'embedding_dim': embeddings.shape[1],
        'num_reference_embeddings': len(embs)
    }
    
    return stats


def adaptive_threshold_selection(embs: List[np.ndarray], 
                               conservative_factor: float = 0.5) -> float:
    """
    Automatically select similarity threshold based on reference embeddings.
    
    Args:
        embs: List of reference embeddings
        conservative_factor: How conservative to be (0.0 = mean, 1.0 = min)
        
    Returns:
        Recommended similarity threshold
    """
    stats = compute_speaker_embedding_statistics(embs)
    
    if not stats:
        return 0.7  # Default threshold
    
    # Use mean - conservative_factor * std as threshold
    # This ensures we're conservative enough to avoid false positives
    mean_sim = stats['mean_intra_similarity']
    std_sim = stats['std_intra_similarity']
    
    threshold = mean_sim - conservative_factor * std_sim
    
    # Ensure threshold is within reasonable bounds
    threshold = max(0.5, min(0.9, threshold))
    
    return threshold




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