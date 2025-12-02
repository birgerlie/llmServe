"""
Embedding service for NB-BERT and NB-SBERT models.

Provides deterministic, semantically-rich embeddings for Norwegian text,
optimized for RAG systems, search, clustering, and knowledge graphs.
"""

import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from app.config import Settings


class TextPreprocessor:
    """
    Text preprocessor for cleaning and normalizing input.

    Removes noise while preserving semantic content:
    - Removes courtesy phrases
    - Removes marketing text
    - Removes UI/meta text
    - Preserves entities, actions, relationships
    """

    # Patterns to remove (Norwegian courtesy/noise phrases)
    NOISE_PATTERNS = [
        r"(?i)\bvennligst\b",  # Please
        r"(?i)\bmvh\.?\b",  # Med vennlig hilsen
        r"(?i)\bmed vennlig hilsen\b",
        r"(?i)\bhi[lsen]*\b",
        r"(?i)\bkontakt(?:[\s:]+[\w@.]+)*",  # Contact info patterns
        r"(?i)\btel(?:efon)?[\s:]*[\d\s+-]+",  # Phone numbers
        r"(?i)\be-?post[\s:]*[\w@.-]+",  # Email patterns
        r"(?i)\bwww\.[\w.-]+",  # URLs
        r"(?i)\bhttps?://[\w./%-]+",  # Full URLs
        r"[\U0001F300-\U0001F9FF]",  # Emojis
        r"[^\w\sæøåÆØÅ.,;:!?()-]",  # Special characters (keep Norwegian chars)
    ]

    @classmethod
    def preprocess(cls, text: str, remove_noise: bool = True) -> str:
        """
        Preprocess text for embedding generation.

        Args:
            text: Input text to preprocess
            remove_noise: Whether to remove noise patterns

        Returns:
            Cleaned text ready for embedding
        """
        if not text or not text.strip():
            return ""

        processed = text.strip()

        if remove_noise:
            for pattern in cls.NOISE_PATTERNS:
                processed = re.sub(pattern, " ", processed)

        # Normalize whitespace
        processed = re.sub(r"\s+", " ", processed).strip()

        return processed


class EmbeddingService:
    """
    Service for generating embeddings using NB-SBERT and NB-BERT models.

    Optimized for:
    - MacBook Air M1 (CPU mode)
    - RAG pipelines
    - Norwegian text (bokmål, nynorsk, dialects, historical)

    Principles:
    - Deterministic output
    - Semantically focused
    - No hallucination
    - 100% consistent across requests
    """

    def __init__(self, settings: Settings):
        """
        Initialize the embedding service.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.device = self._get_device()
        self._sbert_model: Optional[SentenceTransformer] = None
        self._bert_model: Optional[AutoModel] = None
        self._bert_tokenizer: Optional[AutoTokenizer] = None
        self.preprocessor = TextPreprocessor()

        logger.info(f"EmbeddingService initialized with device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device."""
        requested_device = self.settings.device

        if requested_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif requested_device == "mps" and torch.backends.mps.is_available():
            return "mps"
        else:
            if requested_device != "cpu":
                logger.warning(
                    f"Requested device '{requested_device}' not available, falling back to CPU"
                )
            return "cpu"

    def load_sbert_model(self) -> None:
        """Load the NB-SBERT model for sentence embeddings."""
        if self._sbert_model is not None:
            logger.debug("NB-SBERT model already loaded")
            return

        logger.info(f"Loading NB-SBERT model: {self.settings.sbert_model_name}")
        try:
            self._sbert_model = SentenceTransformer(
                self.settings.sbert_model_name,
                device=self.device,
            )
            logger.info("NB-SBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NB-SBERT model: {e}")
            raise

    def load_bert_model(self) -> None:
        """Load the NB-BERT model for encoder representations."""
        if self._bert_model is not None:
            logger.debug("NB-BERT model already loaded")
            return

        logger.info(f"Loading NB-BERT model: {self.settings.bert_model_name}")
        try:
            self._bert_tokenizer = AutoTokenizer.from_pretrained(
                self.settings.bert_model_name
            )
            self._bert_model = AutoModel.from_pretrained(
                self.settings.bert_model_name
            )
            self._bert_model.to(self.device)
            self._bert_model.eval()
            logger.info("NB-BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NB-BERT model: {e}")
            raise

    @property
    def sbert_loaded(self) -> bool:
        """Check if NB-SBERT model is loaded."""
        return self._sbert_model is not None

    @property
    def bert_loaded(self) -> bool:
        """Check if NB-BERT model is loaded."""
        return self._bert_model is not None and self._bert_tokenizer is not None

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """L2-normalize a vector or batch of vectors."""
        if vector.ndim == 1:
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
            return vector
        else:
            norms = np.linalg.norm(vector, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)
            return vector / norms

    def _get_empty_embedding(self, normalize: bool = True) -> List[float]:
        """Return embedding for empty/invalid input."""
        # Return zero vector for empty input
        vector = np.zeros(768, dtype=np.float32)
        return vector.tolist()

    def embed(
        self,
        text: str,
        normalize: bool = True,
        preprocess: bool = True,
    ) -> Tuple[List[float], int]:
        """
        Generate sentence embedding using NB-SBERT.

        Args:
            text: Input text
            normalize: Whether to L2-normalize output
            preprocess: Whether to preprocess text

        Returns:
            Tuple of (embedding vector, dimension)
        """
        if not self.sbert_loaded:
            self.load_sbert_model()

        # Handle empty input
        if not text or not text.strip():
            return self._get_empty_embedding(normalize), 768

        # Preprocess text
        if preprocess:
            text = self.preprocessor.preprocess(text)
            if not text:
                return self._get_empty_embedding(normalize), 768

        # Generate embedding
        embedding = self._sbert_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        return embedding.tolist(), 768

    def embed_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        preprocess: bool = True,
    ) -> Tuple[List[List[float]], int]:
        """
        Generate batch sentence embeddings using NB-SBERT.

        Args:
            texts: List of input texts
            normalize: Whether to L2-normalize output
            preprocess: Whether to preprocess texts

        Returns:
            Tuple of (list of embedding vectors, dimension)
        """
        if not self.sbert_loaded:
            self.load_sbert_model()

        # Handle empty input
        if not texts:
            return [], 768

        # Preprocess texts
        if preprocess:
            processed_texts = [self.preprocessor.preprocess(t) for t in texts]
        else:
            processed_texts = texts

        # Generate embeddings
        embeddings = self._sbert_model.encode(
            processed_texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            batch_size=self.settings.max_batch_size,
        )

        return embeddings.tolist(), 768

    def encode(
        self,
        text: str,
        output_type: str = "pooled_vector",
        normalize: bool = True,
        preprocess: bool = True,
        return_tokens: bool = False,
    ) -> dict:
        """
        Generate encoder representation using NB-BERT.

        Args:
            text: Input text
            output_type: Output type ('vector', 'token_vectors', 'pooled_vector')
            normalize: Whether to L2-normalize output
            preprocess: Whether to preprocess text
            return_tokens: Whether to return tokenized text

        Returns:
            Dictionary with embeddings and metadata
        """
        if not self.bert_loaded:
            self.load_bert_model()

        # Handle empty input
        if not text or not text.strip():
            return {
                "pooled_embedding": self._get_empty_embedding(normalize),
                "token_embeddings": None,
                "tokens": None,
                "dimension": 768,
            }

        # Preprocess text
        if preprocess:
            text = self.preprocessor.preprocess(text)
            if not text:
                return {
                    "pooled_embedding": self._get_empty_embedding(normalize),
                    "token_embeddings": None,
                    "tokens": None,
                    "dimension": 768,
                }

        # Tokenize
        inputs = self._bert_tokenizer(
            text,
            return_tensors="pt",
            max_length=self.settings.max_sequence_length,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get tokens if requested
        tokens = None
        if return_tokens:
            tokens = self._bert_tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0].tolist()
            )

        # Generate embeddings
        with torch.no_grad():
            outputs = self._bert_model(**inputs)

        # Extract embeddings based on output type
        last_hidden_state = outputs.last_hidden_state.cpu().numpy()
        pooler_output = outputs.pooler_output.cpu().numpy()

        # Pooled embedding (CLS token or pooler output)
        pooled = pooler_output[0]
        if normalize:
            pooled = self._normalize_vector(pooled)

        result = {
            "pooled_embedding": pooled.tolist(),
            "dimension": 768,
        }

        # Include token embeddings if requested
        if output_type == "token_vectors":
            token_embeds = last_hidden_state[0]
            if normalize:
                token_embeds = self._normalize_vector(token_embeds)
            result["token_embeddings"] = token_embeds.tolist()

        if return_tokens:
            result["tokens"] = tokens

        return result


# Global service instance (lazy-loaded)
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(settings: Settings) -> EmbeddingService:
    """Get or create the embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(settings)
    return _embedding_service
