from .label_audio import main as LabelFolder
from .embed import main as EmbedLabeledVoiceSegments

def LabelAndEmbed():
    """
    Main function to run the embedding process.
    """
    LabelFolder()
    EmbedLabeledVoiceSegments()

__all__ = ["LabelFolder","EmbedLabeledVoiceSegments","LabelAndEmbed"]