import os

import subprocess

def transcribe_audio(
    file_path,
    model="large-v2",
    align_model=None,
    batch_size=None,
    compute_type=None,
    language=None,
    diarize=False,
    highlight_words=False,
    min_speakers=None,
    max_speakers=None,
    hf_token=None,
    extra_args=None
):
    """
    Transcribe audio using WhisperX CLI with flexible options.

    Parameters:
        file_path (str): 
            Path to the input audio file to be transcribed. Supported formats include wav, mp3, etc.

        model (str, default="large-v2"): 
            The Whisper ASR model to use for transcription.
            Options include "base", "small", "medium", "large", "large-v2", etc.
            Larger models are more accurate but require more memory and computation.

        align_model (str, optional):
            The phoneme-based forced alignment model used to precisely align the transcript to the audio, producing word-level timestamps.
            Alignment in WhisperX works by matching the output of the ASR model (transcribed text) to the audio signal using a specialized model 
            (often a wav2vec2 variant). This process assigns accurate timings to each word or phoneme.
            Forced alignment leverages models trained to recognize small units of speech (phonemes), 
            mapping text segments to corresponding audio sections. This is particularly useful for generating subtitles or timestamped transcripts.
            Common choices include "WAV2VEC2_ASR_LARGE_LV60K_960H" for English. 
            For other languages, WhisperX will attempt to select an appropriate alignment model automatically if not specified.
            For more details, see the WhisperX paper: "WhisperX: Time-Accurate Speech Transcription of Long-Form Audio" (Bain et al., INTERSPEECH 2023)

        batch_size (int, optional): 
            Number of audio segments to process in parallel during inference.
            Increasing batch size speeds up transcription but uses more GPU memory.
            Lower batch size may help on machines with less available memory.

        compute_type (str, optional): 
            Precision type for model inference.
            Common values include "float16" (for faster inference on GPUs with good accuracy),
            and "int8" (for lower memory usage, but may reduce accuracy).
            Use "int8" if you are low on GPU memory or running on CPU.

        language (str, optional): 
            Language code of the audio (e.g., "en" for English, "de" for German).
            This selects the appropriate phoneme alignment model for better timestamp accuracy.
            Useful for multilingual transcription.

        diarize (bool, default=False): 
            If True, enables speaker diarization, partitioning the transcript by speaker identity.
            Requires a HuggingFace access token (see `hf_token`).
            Useful for multi-speaker recordings.

        highlight_words (bool, default=False): 
            If True, highlights word-level timings in the output subtitle (.srt) file.
            Helps visualize precise word alignment in subtitles.

        min_speakers (int, optional): 
            Minimum number of speakers to expect in speaker diarization.
            If known, setting this improves diarization accuracy.

        max_speakers (int, optional): 
            Maximum number of speakers to expect in speaker diarization.
            If known, setting this improves diarization accuracy.

        hf_token (str, optional): 
            HuggingFace access token for downloading speaker diarization models.
            Required if `diarize=True`.

        extra_args (list of str, optional): 
            Additional command-line arguments to pass to WhisperX.
            Useful for advanced usage not explicitly covered by other parameters.

    Returns:
        None. (Runs WhisperX CLI. Output files are saved according to WhisperX defaults.)
    """
    cmd = ["uvx", "whisperx", file_path]
    cmd += ["--model", model]
    
    if align_model:
        cmd += ["--align_model", align_model]
    if batch_size:
        cmd += ["--batch_size", str(batch_size)]
    if compute_type:
        cmd += ["--compute_type", compute_type]
    if language:
        cmd += ["--language", language]
    if diarize:
        cmd += ["--diarize"]
    if highlight_words:
        cmd += ["--highlight_words", "True"]
    if min_speakers:
        cmd += ["--min_speakers", str(min_speakers)]
    if max_speakers:
        cmd += ["--max_speakers", str(max_speakers)]
    if hf_token:
        cmd += ["--hf_token", hf_token]
    if extra_args:
        cmd += extra_args
    
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

# Example usage:
# transcribe_audio("audio.wav", model="large-v2", diarize=True, highlight_words=True, batch_size=4)