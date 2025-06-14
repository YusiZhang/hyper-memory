"""
Embedding utilities for generating text embeddings using OpenAI API.
"""

import os
from typing import List
import openai
from openai import OpenAI


def embed(text: str) -> List[float]:
    """
    Generate embeddings for text using OpenAI text-embedding-3-small model.
    
    Args:
        text: Input text to embed
        
    Returns:
        List of floats representing the embedding vector (1536 dimensions)
        
    Raises:
        Exception: If embedding generation fails
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Validate input
        if not text or not text.strip():
            raise Exception("Text cannot be empty")
        
        # Generate embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text.strip()
        )
        
        # Extract embedding vector
        embedding = response.data[0].embedding
        
        # Validate embedding dimensions
        if len(embedding) != 1536:
            raise Exception(f"Unexpected embedding dimension: {len(embedding)}, expected 1536")
        
        return embedding
        
    except openai.OpenAIError as e:
        raise Exception(f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise Exception(f"Embedding generation failed: {str(e)}")


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a single API call.
    
    Args:
        texts: List of input texts to embed
        
    Returns:
        List of embedding vectors
        
    Raises:
        Exception: If embedding generation fails
    """
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Validate input
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise Exception("No valid texts provided")
        
        # Generate embeddings
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=valid_texts
        )
        
        # Extract embedding vectors
        embeddings = [data.embedding for data in response.data]
        
        # Validate all embeddings have correct dimensions
        for i, embedding in enumerate(embeddings):
            if len(embedding) != 1536:
                raise Exception(f"Unexpected embedding dimension for text {i}: {len(embedding)}, expected 1536")
        
        return embeddings
        
    except openai.OpenAIError as e:
        raise Exception(f"OpenAI API error: {str(e)}")
    except Exception as e:
        raise Exception(f"Batch embedding generation failed: {str(e)}")