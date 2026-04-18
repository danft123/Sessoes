# LLM

LLM-powered analysis of transcriptions using Anthropic Claude.

## Overview

This module provides functions to process transcriptions through Claude (Sonnet 4) for various analysis tasks:
- Summary generation
- Progress report extraction
- Idea capture with tags

## Installation

```bash
pip install langchain-anthropic python-dotenv
```

Create a `.env` file with your Anthropic API key:
```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Basic Query

```python
from src.LLM.utils import query

response = query("What is the capital of France?")
```

### Summary Generator

```python
from src.LLM.utils import transcription_sumary

with open("data/transcripts/session.txt", "r") as f:
    transcription = f.read()

summary = transcription_sumary(transcription)
```

The summary includes source timestamps in format `[start - end]` for specific references.

### Progress Report Extractor

```python
from src.LLM.utils import transcription_progress_report

report = transcription_progress_report(transcription)
```

Extracts key points, insights, and actions from the transcription.

### Idea Capture (Blobs)

```python
from src.LLM.utils import transcription_capture_tags

blobs = transcription_capture_tags(transcription)
```

Extracts "blobs" - self-contained ideas the user wants to save.

**Supported Tags:**
- `[progress_report]` - Current progress updates
- `[action_item]` - Tasks or follow-ups
- `[key_decision]` - Important decisions made
- `[new_idea]` - New concepts or suggestions

## Function Reference

### `query(prompt: str) -> str`

Low-level function to query the LLM directly.

### `transcription_sumary(transcription: str) -> str`

Interprets transcription and provides a concise summary with source timestamps.

**Prompt Characteristics:**
- Focuses on main ideas and context
- Handles transcription inaccuracies gracefully
- Provides timestamps for specific references

### `transcription_progress_report(transcription: str) -> str`

Extracts progress-focused content from transcriptions.

### `transcription_capture_tags(transcription: str) -> str`

Identifies and extracts tagged "blobs of ideas" from the transcription.

**Trigger Phrase:** "capture" or "capture a blob of idea"

**Blob Structure:**
- **Name:** Short descriptive title
- **Description:** Detailed explanation

## Configuration

The module initializes with:
```python
llm = ChatAnthropic(model='claude-sonnet-4-20250514')
```

Default paths in `__main__`:
```python
transcripts_dir = 'data/transcripts'
summaries_dir = 'data/transcript_summaries'
```

## Batch Processing

Run as script to process all transcripts:

```bash
python -m src.LLM.utils
```

This generates summaries for all `.txt` files in `data/transcripts/` and saves them to `data/transcript_summaries/`.

## Error Handling

The module handles:
- Transcription inaccuracies (halucinations)
- Missing timestamps
- Empty transcriptions