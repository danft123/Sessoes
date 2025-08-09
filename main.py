# from src.Voice.MaskAndTranscript import MaskAndTranscriptMain
# from src.Voice.Embedding import LabelFolder, LabelAndEmbed, EmbedLabeledVoiceSegments

# #MaskAndTranscriptMain()  # Call the main function to start the process
# LabelAndEmbed()
# LabelFolder()  # Call the labeling function to label audio segments
# EmbedLabeledVoiceSegments()

from src.Voice.Transcript.whisperx import transcribe_audio, process_audio_folder

pathfolder = "data/Raw/ProcessedAudios/TEST"
output_dir = "data/Raw/ProcessedAudios/TEST/Output"

process_audio_folder(pathfolder, output_dir=output_dir)  # Process all audio files in the specified folder