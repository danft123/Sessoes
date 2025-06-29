from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='/home/danft/Documents/Pessoal/Sessões/.env')

llm = ChatAnthropic(model = 'claude-sonnet-4-20250514')



def query(prompt: str) -> str:
    """
    Queries the LLM with the given prompt and returns the response.
    
    Args:
        prompt (str): The prompt to send to the LLM.
        
    Returns:
        str: The response from the LLM.
    """
    response = llm.invoke(prompt)
    return response.content if response else ""

def transcription_sumary(transcription: str) -> str:
    """
    Interprets the transcription using the LLM.
    
    Args:
        transcription (str): The transcription text to interpret.
        
    Returns:
        str: The interpreted result from the LLM.
    """
    PROMPT_ROLE = """You are a helpful assistant that interprets transcriptions of audio recordings."""
    PROMPT_TASK = """Interpret the transcription and provide a concise summary of its content. You must always provide source by mencioning the timestamp when commenting on a specific part of the transcription. If you are not sure about the timestamp, use the format [start - end] where start and end are the timestamps in seconds. If you are not sure about the content, use [unknown]."""
    PROMPT_OBSERVATIONS = """The transcription may contain multiple segments, each with a start and end timestamp.
    Your task is to analyze the transcription and summarize the key points, insights, or actions that can be derived from it.
    Focus on the main ideas and any important details that stand out in the transcription."""
    PROMPT_CAUTION = """There may be halucinations or inaccuracies in the transcription in the form of random words or phrases. Try to grasp the main ideas and context, but be aware that some parts may not make sense or may not be relevant to the overall content."""

    PROMPT = f"""{PROMPT_ROLE}
    # TASK
    {PROMPT_TASK}
    # OBSERVATIONS
    {PROMPT_OBSERVATIONS}
    # CAUTION
    {PROMPT_CAUTION}
    # INPUT

    """
    prompt = PROMPT + transcription.strip()
    return query(prompt)

def transcription_progress_report(transcription: str) -> str:
    """
    Interprets the transcription using the LLM and extracts the progress report.
    
    Args:
        transcription (str): The transcription text to interpret.
        
    Returns:
        str: The interpreted result from the LLM.
    """
    PROMPT_ROLE = """You are a helpful assistant that interprets transcriptions of audio recordings."""
    PROMPT_TASK = """Your task is to analyze the transcription and extract a progress report. The progress report should include key points, insights, or actions that can be derived from the transcription. Focus on the main ideas and any important details that stand out in the transcription."""
    PROMPT_OBSERVATIONS = """The transcription may contain multiple segments, each with a start and end timestamp.
    Your task is to analyze the transcription and focus entirely on progress reports.
    Focus on the main ideas and any important details that stand out in the transcription."""
    PROMPT_CAUTION = """There may be halucinations or inaccuracies in the transcription in the form of random words or phrases. Try to grasp the main ideas and context, but be aware that some parts may not make sense or may not be relevant to the overall content."""

    PROMPT = f"""{PROMPT_ROLE}
    # TASK
    {PROMPT_TASK}
    # OBSERVATIONS
    {PROMPT_OBSERVATIONS}
    # CAUTION
    {PROMPT_CAUTION}
    # INPUT

    """
    prompt = PROMPT + transcription.strip()
    return query(prompt)

def transcription_capture_tags(transcription: str) -> str:
    """
    Interprets the transcription using the LLM and extracts special tags.
    
    Args:
        transcription (str): The transcription text to interpret.
        
    Returns:
        str: The interpreted result from the LLM.
    """

    PROMPT_ROLE = "AI assistant specializing in analyzing audio transcriptions to identify and extract key ideas."
    PROMPT_CORE_TASK = "Your primary function is to identify 'blobs of ideas' within a transcription. A 'blob of idea' is a self-contained concept or thought the user wants to save."
    PROMPT_TRIGGER_PHRASE = "You must act when you detect the user saying 'capture' or 'capture a blob of idea,' which will be immediately followed by the content of the blob."
    PROMPT_BLOB_STRUCTURE_NAME = "A short, descriptive title that summarizes the core idea."
    PROMPT_BLOB_STRUCTURE_DESCRIPTION = "A more detailed explanation of the concept."
    PROMPT_SPECIAL_TAGS = "[progress_report], [action_item], [key_decision], [new_idea]"
    PROMPT_CURRENT_ASSIGNMENT_FOCUS_TAG = "[progress_report]"
    PROMPT_IMPORTANT_CONSIDERATIONS_TRANSCRIPTION_QUALITY = "The transcription may contain inaccuracies or nonsensical phrases. Focus on the overall context and the main points to understand the user's intent."
    PROMPT_IMPORTANT_CONSIDERATIONS_TIMESTAMPS = "The transcription will include start and end timestamps for different segments, which you can use for context but do not need to include in the output."

    PROMPT = f"""
    Role: {PROMPT_ROLE}
    Core Task: {PROMPT_CORE_TASK}
    Trigger Phrase: {PROMPT_TRIGGER_PHRASE}
    Blob of Idea Structure:
    Each captured blob must have two parts:
    Name: {PROMPT_BLOB_STRUCTURE_NAME}
    Description: {PROMPT_BLOB_STRUCTURE_DESCRIPTION}
    Special Tags:
    Blobs can be categorized with special tags, which the user will state. Your instructions will specify which tag to focus on. Common tags include:
    {PROMPT_SPECIAL_TAGS}
    Your Current Assignment:
    For the upcoming transcription, your exclusive focus is to identify and list all blobs tagged as {PROMPT_CURRENT_ASSIGNMENT_FOCUS_TAG}.
    Important Considerations:
    Transcription Quality: {PROMPT_IMPORTANT_CONSIDERATIONS_TRANSCRIPTION_QUALITY}
    Timestamps: {PROMPT_IMPORTANT_CONSIDERATIONS_TIMESTAMPS}
    """
    prompt = PROMPT + transcription.strip()
    return query(prompt)

if __name__ == "__main__":
    transcripts_dir = '/home/danft/Documents/Pessoal/Sessões/data/transcripts'
    summaries_dir = '/home/danft/Documents/Pessoal/Sessões/data/transcript_summaries'
    os.makedirs(summaries_dir, exist_ok=True)

    for filename in os.listdir(transcripts_dir):
        if filename.endswith('.txt'):
            summary_filename = filename.replace('.txt', '_summary.txt')
            summary_path = os.path.join(summaries_dir, summary_filename)
            if os.path.exists(summary_path):
                print(f"Summary already exists for {filename}, skipping...")
                continue
            transcript_path = os.path.join(transcripts_dir, filename)
            with open(transcript_path, 'r') as f:
                transcription = f.read()
            output = transcription_sumary(transcription)
            print(f"Summary for {filename}:\n{output}\n")

            with open(summary_path, 'w') as f:
                f.write(output)
            print(f"Summary saved to {summary_path}")