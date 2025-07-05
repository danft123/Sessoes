import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
# classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
# signal, fs =torchaudio.load('tests/samples/ASR/spk1_snt1.wav')
# embeddings = classifier.encode_batch(signal)


import os
from src.Voice.MaskAndTranscript.voice_mask import create_voice_mask, create_combined_audio
import conf.config as config # Import the new config file

import librosa
import soundfile as sf
import numpy as np
from typing import List, Tuple, Dict
import tempfile
import os
import pygame
from pygame import mixer
import webrtcvad
import uuid
SAMPLE_RATE = config.SAMPLE_RATE


def label_voice_segments_interactively(
    audio_file: str,
    original_timestamps: List[Tuple[float, float]],
    sample_rate: int = SAMPLE_RATE,
    vad_aggressiveness_negatives: int = 3  # From conf.VAD_AGGRESSIVENESS_NEGATIVES
) -> Tuple[List[Dict], List[Dict]]:
    """
    Interactively label voice segments by playing each one and asking user for confirmation and name.
    Also processes negative segments through VAD filtering and user review.
    
    Args:
        audio_file: Path to the original audio file
        original_timestamps: List of (start, end) timestamps for voice segments
        sample_rate: Audio sample rate
        vad_aggressiveness_negatives: VAD aggressiveness level for negative filtering (0-3)
    
    Returns:
        Tuple of (positive_segments, negative_segments):
        - positive_segments: List of dictionaries with 'start', 'end', 'name', 'duration' for approved isolated segments
        - negative_segments: List of dictionaries with 'start', 'end', 'reason', 'duration' for rejected segments
    """
    # Load the original audio
    audio, sr = librosa.load(audio_file, sr=sample_rate)
    
    # Initialize pygame mixer for audio playback
    mixer.init(frequency=sample_rate, size=-16, channels=1, buffer=512)
    
    # Initialize VAD for negative filtering
    vad = webrtcvad.Vad(vad_aggressiveness_negatives)
    
    labeled_segments = []
    negative_segments = []
    segments_for_negative_review = []
    
    print(f"Found {len(original_timestamps)} voice segments to review.")
    print("Instructions:")
    print("- Each segment will be played automatically")
    print("- Type 'y' if it's an isolated voice segment, 'n' to skip")
    print("- For approved segments, you'll be asked to enter the speaker's name")
    print("- Press Enter to replay the current segment")
    print("- Type 'q' to quit labeling")
    print("-" * 50)
    
    # First pass: Process positive segments and collect negatives
    for i, (start, end) in enumerate(original_timestamps):
        print(f"\nSegment {i+1}/{len(original_timestamps)} ({start:.2f}s - {end:.2f}s)")
        
        # Extract the segment
        start_idx, end_idx = int(start * sr), int(end * sr)
        segment = audio[start_idx:end_idx]
        
        # Create temporary file for playback
        with tempfile.NamedTemporaryFile(suffix='.wav', delete = False) as temp_file:
            temp_filename = temp_file.name
            sf.write(temp_filename, segment, sample_rate)
        
        try:
            # Play the segment
            def play_segment():
                try:
                    mixer.music.load(temp_filename)
                    mixer.music.play()
                    print("Playing segment...")
                except Exception as e:
                    print(f"Error playing audio: {e}")
            
            play_segment()
            
            while True:
                response = input("Is this an isolated voice segment? (y/n/Enter=replay/q=quit): ").strip().lower()
                
                if response == 'q':
                    print("Quitting labeling process...")
                    mixer.quit()
                    return labeled_segments, negative_segments
                
                elif response == '':
                    # Replay the segment
                    play_segment()
                    continue
                
                elif response == 'y':
                    # Ask for speaker name
                    while True:
                        speaker_name = input("Enter speaker name: ").strip()
                        if speaker_name:
                            labeled_segments.append({
                                'start': start,
                                'end': end,
                                'name': speaker_name,
                                'duration': end - start
                            })
                            print(f"✓ Segment labeled as '{speaker_name}'")
                            break
                        else:
                            print("Please enter a valid name.")
                    break
                
                elif response == 'n':
                    print("✗ Segment skipped - will be processed as negative")
                    # Store for negative processing
                    segments_for_negative_review.append((start, end, segment))
                    break
                
                else:
                    print("Please enter 'y', 'n', Enter to replay, or 'q' to quit.")
        
        finally:
            # Clean up temporary file
            try:
                mixer.music.stop()
                os.unlink(temp_filename)
            except:
                pass
    
    # Second pass: Process negative segments through VAD
    if segments_for_negative_review:
        print(f"\nProcessing {len(segments_for_negative_review)} negative segments through VAD...")
        
        for start, end, segment in segments_for_negative_review:
            # Check if segment contains speech using VAD
            has_speech = check_speech_in_segment(segment, vad, sample_rate)
            
            if not has_speech:
                # No speech detected, automatically categorize as filtered
                negative_segments.append({
                    'start': start,
                    'end': end,
                    'reason': 'filtered_vad_negatives',
                    'duration': end - start
                })
            else:
                # Speech detected, needs user review for reason
                print(f"\nNegative segment {start:.2f}s - {end:.2f}s (contains speech)")
                
                # Create temporary file for playback
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_filename = temp_file.name
                    sf.write(temp_filename, segment, sample_rate)
                
                try:
                    # Play the segment
                    def play_negative_segment():
                        try:
                            mixer.music.load(temp_filename)
                            mixer.music.play()
                            print("Playing negative segment...")
                        except Exception as e:
                            print(f"Error playing audio: {e}")
                    
                    play_negative_segment()
                    
                    while True:
                        response = input("Enter reason for rejection (Enter=replay/q=quit): ").strip()
                        
                        if response.lower() == 'q':
                            print("Quitting negative labeling process...")
                            mixer.quit()
                            return labeled_segments, negative_segments
                        
                        elif response == '':
                            # Replay the segment
                            play_negative_segment()
                            continue
                        
                        elif response:
                            negative_segments.append({
                                'start': start,
                                'end': end,
                                'reason': response,
                                'duration': end - start
                            })
                            print(f"✓ Negative segment labeled with reason: '{response}'")
                            break
                        
                        else:
                            print("Please enter a reason or press Enter to replay.")
                
                finally:
                    # Clean up temporary file
                    try:
                        mixer.music.stop()
                        os.unlink(temp_filename)
                    except:
                        pass
    
    # Clean up pygame mixer
    mixer.quit()
    
    # Print summary
    print(f"\nLabeling complete!")
    print(f"Approved {len(labeled_segments)} isolated voice segments:")
    for segment in labeled_segments:
        print(f"- {segment['name']}: {segment['start']:.2f}s - {segment['end']:.2f}s ({segment['duration']:.2f}s)")
    
    print(f"\nProcessed {len(negative_segments)} negative segments:")
    for segment in negative_segments:
        print(f"- {segment['reason']}: {segment['start']:.2f}s - {segment['end']:.2f}s ({segment['duration']:.2f}s)")
    
    return labeled_segments, negative_segments


def check_speech_in_segment(segment: np.ndarray, vad: webrtcvad.Vad, sample_rate: int) -> bool:
    """
    Check if a segment contains speech using WebRTC VAD.
    
    Args:
        segment: Audio segment as numpy array
        vad: WebRTC VAD instance
        sample_rate: Sample rate of the audio
    
    Returns:
        True if speech is detected, False otherwise
    """
    # WebRTC VAD works with specific frame sizes and sample rates
    # Frame length in samples (10ms, 20ms, or 30ms frames are supported)
    frame_duration_ms = 20  # 20ms frames
    frame_length = int(sample_rate * frame_duration_ms / 1000)
    
    # Ensure we have enough samples
    if len(segment) < frame_length:
        return False
    
    speech_frames = 0
    total_frames = 0
    
    # Process audio in frames
    for i in range(0, len(segment) - frame_length, frame_length):
        frame = segment[i:i + frame_length]
        
        # Convert to int16 format required by VAD
        frame_bytes = (frame * 32767).astype(np.int16).tobytes()
        
        try:
            is_speech = vad.is_speech(frame_bytes, sample_rate)
            if is_speech:
                speech_frames += 1
            total_frames += 1
        except Exception as e:
            # If VAD fails on this frame, skip it
            continue
    
    # Consider segment as containing speech if more than 10% of frames contain speech
    if total_frames == 0:
        return False
    
    speech_ratio = speech_frames / total_frames
    return speech_ratio > 0.1


def save_labeled_segments(labeled_segments: List[Dict], filepath: str, output_file: str = "labeled_segments.json"):
    """
    Save labeled segments to a JSON file for later use.
    """
    import json
    final_output = {'filepath': filepath, 'segments': labeled_segments}
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)
    
    print(f"Labeled segments saved to {output_file}")

# Alternative implementation without pygame (using system audio player)
# def label_voice_segments_system_player(
#     audio_file: str,
#     original_timestamps: List[Tuple[float, float]],
#     sample_rate: int = SAMPLE_RATE
# ) -> List[Dict]:
#     """
#     Alternative implementation using system audio player (works on most systems).
#     """
#     import subprocess
#     import platform
    
#     # Load the original audio
#     audio, sr = librosa.load(audio_file, sr=sample_rate)
    
#     labeled_segments = []
    
#     print(f"Found {len(original_timestamps)} voice segments to review.")
#     print("Instructions:")
#     print("- Each segment will be saved and played using your system's default audio player")
#     print("- Type 'y' if it's an isolated voice segment, 'n' to skip")
#     print("- For approved segments, you'll be asked to enter the speaker's name")
#     print("- Type 'r' to replay the current segment")
#     print("- Type 'q' to quit labeling")
#     print("-" * 50)
    
#     # Determine system audio player command
#     system = platform.system().lower()
#     if system == 'darwin':  # macOS
#         play_cmd = ['afplay']
#     elif system == 'linux':
#         play_cmd = ['aplay']  # or 'paplay' for PulseAudio
#     elif system == 'windows':
#         play_cmd = ['start', '']  # Windows will use default player
#     else:
#         print("Warning: Unknown system, audio playback might not work")
#         play_cmd = ['play']  # fallback to sox
    
#     for i, (start, end) in enumerate(original_timestamps):
#         print(f"\nSegment {i+1}/{len(original_timestamps)} ({start:.2f}s - {end:.2f}s)")
        
#         # Extract the segment
#         start_idx, end_idx = int(start * sr), int(end * sr)
#         segment = audio[start_idx:end_idx]
        
#         # Create temporary file for playback
#         temp_filename = f"temp_segment_{i}.wav"
#         sf.write(temp_filename, segment, sample_rate)
        
#         def play_segment():
#             try:
#                 if system == 'windows':
#                     subprocess.run(play_cmd + [temp_filename], shell=True, check=False)
#                 else:
#                     subprocess.run(play_cmd + [temp_filename], check=False)
#                 print("Playing segment...")
#             except Exception as e:
#                 print(f"Error playing audio: {e}")
#                 print(f"You can manually play: {temp_filename}")
        
#         try:
#             play_segment()
            
#             while True:
#                 response = input("Is this an isolated voice segment? (y/n/r=replay/q=quit): ").strip().lower()
                
#                 if response == 'q':
#                     print("Quitting labeling process...")
#                     return labeled_segments
                
#                 elif response == 'r':
#                     # Replay the segment
#                     play_segment()
#                     continue
                
#                 elif response == 'y':
#                     # Ask for speaker name
#                     while True:
#                         speaker_name = input("Enter speaker name: ").strip()
#                         if speaker_name:
#                             labeled_segments.append({
#                                 'start': start,
#                                 'end': end,
#                                 'name': speaker_name,
#                                 'duration': end - start
#                             })
#                             print(f"✓ Segment labeled as '{speaker_name}'")
#                             break
#                         else:
#                             print("Please enter a valid name.")
#                     break
                
#                 elif response == 'n':
#                     print("✗ Segment skipped")
#                     break
                
#                 else:
#                     print("Please enter 'y', 'n', 'r' to replay, or 'q' to quit.")
        
#         finally:
#             # Clean up temporary file
#             try:
#                 os.unlink(temp_filename)
#             except:
#                 pass
    
import os
import uuid
import json
from datetime import datetime

def process_audio_file(filepath, output_dir, temp_dir):
    """Processes a single audio file through the full pipeline."""
    print("-" * 50)
    # if os.path.join(output_dir, f"{session_id}_metadata.json") exists then we already processed, return ...
    # if 'art1_change_to_hardware_' in filepath:
    # print(f"Skipping already processed file: {os.path.basename(filepath)}")
    # return ...
    print(f"Processing: {os.path.basename(filepath)}")

    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    session_metadata = {
        "filepath": filepath,
        "filename": os.path.basename(filepath),
        "session_id": session_id,
        "FirstVAD": None,
        "InteractiveLabeling": {
            "positive_segments": [],
            "negative_segments": []
        }
    }

    try:
        print("Step 1: Analyzing voice segments...")
        original_timestamps = create_voice_mask(filepath)
        if not original_timestamps:
            print("No voice segments detected. Skipping file.")
            return

        session_metadata["FirstVAD"] = original_timestamps # Store the original VAD timestamps

        # The create_combined_audio step is not needed for interactive labeling.
        # It can be moved or removed if its only purpose was for this step.
        # If you need the masked file for other reasons, you can leave this part.
        # masked_filepath = os.path.join(temp_dir, f"{os.path.basename(filepath)}_masked.mp3")
        # print("Step 2: Creating temporary masked audio...")
        # create_combined_audio(filepath, original_timestamps, masked_filepath)

        print("Step 2: Labeling voice segments interactively...") # Updated step number
        try:
            # CORRECTED: Pass the original audio file, not the masked one.
            positive_segments, negative_segments = label_voice_segments_interactively(filepath, original_timestamps)
        except (ImportError, NameError) as e: # Catch potential NameError for pygame/mixer
            assert False, "Interactive labeling requires pygame and webrtcvad. Please install them using pip."
            # print(f"Interactive labeling dependency error: {e}. Falling back to system player.")
            # # As a fallback, you can call the system player version
            # positive_segments = label_voice_segments_system_player(filepath, original_timestamps)
            # negative_segments = [] # The system player version doesn't handle negatives

        session_metadata["InteractiveLabeling"]["positive_segments"] = [
            {'start': seg['start'], 'end': seg['end'], 'name': seg['name'], 'duration': seg['duration']}
            for seg in positive_segments
        ]
        session_metadata["InteractiveLabeling"]["negative_segments"] = [
            {'start': seg['start'], 'end': seg['end'], 'reason': seg['reason'], 'duration': seg['duration']}
            for seg in negative_segments
        ]

        # Generate timestamped filename for the combined metadata file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_filename = os.path.join(output_dir, f"sessions_metadata_{timestamp}.json")
        
        # Load existing data if file exists, otherwise create new dict
        if os.path.exists(metadata_filename):
            with open(metadata_filename, 'r') as f:
                all_sessions_data = json.load(f)
        else:
            all_sessions_data = {}
        
        # Add current session data using UUID as key
        all_sessions_data[session_id] = session_metadata
        
        # Save combined metadata to timestamped JSON file
        with open(metadata_filename, 'w') as f:
            json.dump(all_sessions_data, f, indent=4)
        
        print(f"Session metadata saved to: {metadata_filename} with UUID: {session_id}")
        print("Processing complete.")

    except Exception as e:
        print(f"An error occurred during processing: {e}")



def main():
    """Main function to run the batch processing."""
    input_dir = config.INPUT_AUDIO_DIR
    output_dir = config.OUTPUT_LABEL_DIR
    temp_dir = config.TEMP_DIR

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    files_to_process = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp3')]
    
    if not files_to_process:
        print(f"No MP3 files found in '{input_dir}'. Please add audio files to process.")
        return
        
    print(f"Found {len(files_to_process)} MP3 file(s) in '{input_dir}'.")

    for filename in files_to_process:
        filepath = os.path.join(input_dir, filename)
        process_audio_file(filepath, output_dir, temp_dir)
        
    print("\nBatch processing complete.")

if __name__ == '__main__':
    main()