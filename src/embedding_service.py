"""
Embedding service for generating vector embeddings using llama.cpp.

This module provides embedding generation for semantic search using the
mxbai-embed-large-v1 model via llama-cpp-python.
"""

import logging
from typing import List, Optional
from llama_cpp import Llama

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings using llama.cpp.
    
    Uses the mxbai-embed-large-v1 model for high-quality embeddings
    optimized for semantic search and retrieval.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding service.
        
        Args:
            config: Embedding model configuration.
        """
        self.config = config
        self.model: Optional[Llama] = None
        
        logger.info(f"Initializing embedding service with model: {config.model_path}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the embedding model using llama.cpp."""
        try:
            self.model = Llama(
                model_path=self.config.model_path,
                n_ctx=self.config.n_ctx,
                n_gpu_layers=self.config.n_gpu_layers,
                embedding=True,  # Enable embedding mode
                verbose=False
            )
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise RuntimeError(f"Embedding model initialization failed: {e}")
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed.
            
        Returns:
            List[float]: Embedding vector.
            
        Raises:
            RuntimeError: If model is not available or embedding fails.
        """
        if self.model is None:
            raise RuntimeError("Embedding model not available")
        
        try:
            # Generate embedding
            embedding = self.model.embed(text)
            
            # Normalize the embedding (important for cosine similarity)
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]
            
            logger.debug(f"Generated embedding of dimension {len(embedding)} for text: {text[:50]}...")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List[List[float]]: List of embedding vectors.
        """
        embeddings = []
        for text in texts:
            try:
                embedding = self.embed(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to embed text, using zero vector: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 1024)  # mxbai-embed-large-v1 has 1024 dimensions
        
        return embeddings
    
    def is_available(self) -> bool:
        """Check if embedding model is available."""
        return self.model is not None
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        # mxbai-embed-large-v1 produces 1024-dimensional embeddings
        return 1024
    
    def __del__(self):
        """Clean up resources."""
        if self.model is not None:
            logger.info("Cleaning up embedding model")
            del self.model
