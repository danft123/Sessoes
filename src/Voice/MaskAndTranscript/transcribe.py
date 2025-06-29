import torch
import torchaudio
import gc
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Import settings from the central config file
from conf.config import (
    WHISPER_MODEL_ID, DEVICE, TORCH_DTYPE, SAMPLE_RATE, 
    TRANSCRIPTION_LANGUAGE, TRANSCRIPTION_TASK, CHUNK_LENGTH_S, CHUNK_OVERLAP_S
)
from .audio_utils import transcribe_with_progress # Use relative import

print("Loading model and processor...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_MODEL_ID, torch_dtype=TORCH_DTYPE, low_cpu_mem_usage=True, use_safetensors=True
).to(DEVICE)
processor = AutoProcessor.from_pretrained(WHISPER_MODEL_ID)

def transcribe_audio(audio_file_path: str, model = model, processor = processor) -> dict | None:
    """
    Transcribes an audio file using the configured Whisper model.
    """
    print(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")
    
    try:
        audio, sr = torchaudio.load(audio_file_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            audio = resampler(audio)
            
        sample = {"array": audio.squeeze().numpy(), "sampling_rate": SAMPLE_RATE}
        del audio
        gc.collect()

    except FileNotFoundError:
        print(f"Error: Audio file not found at '{audio_file_path}'")
        return None

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
        chunk_length_s=30,
    )

    generate_kwargs = {
        "task": TRANSCRIPTION_TASK,
        "return_timestamps": True,
    }
    
    print("Starting transcription...")
    result = transcribe_with_progress(
        sample, pipe, generate_kwargs, DEVICE, CHUNK_LENGTH_S, CHUNK_OVERLAP_S
    )

    # Clean up memory
    del model, processor, pipe
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return result