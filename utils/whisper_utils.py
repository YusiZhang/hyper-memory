"""
Whisper utilities for speech-to-text transcription using OpenAI API.
"""

import os
import tempfile
from typing import Optional
import openai
from openai import OpenAI


def transcribe(wav_path: str) -> str:
    """
    Transcribe audio file to text using OpenAI Whisper.
    
    Args:
        wav_path: Path to the audio file (WAV format)
        
    Returns:
        Transcribed text as string
        
    Raises:
        Exception: If transcription fails or file is too large (>25MB)
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Check file size (OpenAI has 25MB limit)
        file_size = os.path.getsize(wav_path)
        if file_size > 25 * 1024 * 1024:  # 25MB in bytes
            raise Exception("Audio file is too large (>25MB). Please record a shorter audio.")
        
        # Open and transcribe the audio file
        with open(wav_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        return transcript.strip() if transcript else ""
        
    except openai.OpenAIError as e:
        raise Exception(f"OpenAI API error: {str(e)}")
    except FileNotFoundError:
        raise Exception(f"Audio file not found: {wav_path}")
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")


def transcribe_audio_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transcribe audio from bytes by creating a temporary file.
    
    Args:
        audio_bytes: Audio data as bytes
        filename: Temporary filename to use
        
    Returns:
        Transcribed text as string
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name
        
        # Transcribe the temporary file
        result = transcribe(temp_path)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return result
        
    except Exception as e:
        # Clean up temporary file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise e