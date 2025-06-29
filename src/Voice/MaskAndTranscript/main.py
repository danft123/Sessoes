import os
import json
import datetime
from src.Voice.MaskAndTranscript.voice_mask import create_voice_mask, create_combined_audio, export_audio_segments
from src.Voice.MaskAndTranscript.transcribe import transcribe_audio
from src.Voice.MaskAndTranscript.audio_utils import remap_timestamps
import conf.config as config # Import the new config file

def format_timestamp(seconds):
    """Formats seconds into MM:SS.ss format, handling None."""
    if seconds is None:
        return "N/A"
    return f"{int(seconds//60):02d}:{seconds%60:05.2f}"

def save_clean_transcript(final_data, output_filepath):
    """Saves a human-readable .txt transcript."""
    clean_filepath = output_filepath.replace('_transcript.json', '_clean.txt')
    with open(clean_filepath, 'w', encoding='utf-8') as f:
        metadata = final_data.get("metadata", {})
        f.write("=== AUDIO TRANSCRIPT ===\n")
        f.write(f"File: {metadata.get('original_filename', 'N/A')}\n")
        f.write(f"Date: {metadata.get('processing_date_utc', 'N/A')}\n")
        f.write("="*50 + "\n\n")
        
        for chunk in final_data.get("chunks", []):
            start = format_timestamp(chunk.get("timestamp_start_seconds"))
            end = format_timestamp(chunk.get("timestamp_end_seconds"))
            f.write(f"[{start} - {end}]: {chunk['text']}\n")
            
    print(f"Clean transcript saved to: {clean_filepath}")

def process_audio_file(filepath, output_dir, temp_dir):
    """Processes a single audio file through the full pipeline."""
    print("-" * 50)
    print(f"Processing: {os.path.basename(filepath)}")
    
    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)
    
    masked_filepath = os.path.join(temp_dir, f"{os.path.basename(filepath)}_masked.mp3")
    masked_folder = os.path.dirname(masked_filepath)

    try:
        print("Step 1: Analyzing voice segments...")
        original_timestamps = create_voice_mask(filepath)
        if not original_timestamps:
            print("No voice segments detected. Skipping file.")
            return

        print("Step 2: Creating temporary masked audio...")
        create_combined_audio(filepath, original_timestamps, masked_filepath)

        # print("Step 2: Creating multiple files...")
        # export_audio_segments(filepath, original_timestamps, masked_folder)
        
        print("Step 3: Transcribing masked audio...")
        transcription_result = transcribe_audio(masked_filepath)
        if not transcription_result:
            raise Exception("Transcription failed.")

        print("Step 4: Remapping timestamps...")
        final_result = remap_timestamps(transcription_result, original_timestamps)

        print("Step 5: Preparing final output...")
        output_data = {
            "metadata": {
                "original_filename": os.path.basename(filepath),
                "processing_date_utc": datetime.datetime.utcnow().isoformat(),
                "model_used": config.WHISPER_MODEL_ID,
            },
            "transcription_full": transcription_result.get("text"),
            "chunks": [
                {
                    "timestamp_start_seconds": chunk["real_timestamp"][0],
                    "timestamp_end_seconds": chunk["real_timestamp"][1],
                    "text": chunk['text'].strip()
                } for chunk in final_result.get("remapped_chunks", [])
            ]
        }
        
        output_filename = os.path.basename(filepath).replace('.mp3', '_transcript.json')
        output_filepath = os.path.join(output_dir, output_filename)
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved transcript to: {output_filepath}")

        save_clean_transcript(output_data, output_filepath)

    except Exception as e:
        print(f"An error occurred while processing {filepath}: {e}")
    finally:
        if os.path.exists(masked_filepath):
            # os.remove(masked_filepath)
            print(f"Cleaned up temporary file: {masked_filepath}")
            
    print("-" * 50)

def main():
    """Main function to run the batch processing."""
    input_dir = config.INPUT_AUDIO_DIR
    output_dir = config.OUTPUT_TRANSCRIPT_DIR
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