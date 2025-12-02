"""
gRPC client for NB-BERT/NB-SBERT embeddings.

Example client for connecting to the embedding service over Thunderbolt.
"""

from typing import Iterator, List, Optional, Tuple

import grpc
import numpy as np
from loguru import logger

# Import generated gRPC code
try:
    from app.grpc.generated import embedding_pb2
    from app.grpc.generated import embedding_pb2_grpc
except ImportError:
    embedding_pb2 = None
    embedding_pb2_grpc = None


# Client options optimized for Thunderbolt connections
GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 100 * 1024 * 1024),
    ("grpc.max_receive_message_length", 100 * 1024 * 1024),
    ("grpc.keepalive_time_ms", 10000),
    ("grpc.keepalive_timeout_ms", 5000),
    ("grpc.keepalive_permit_without_calls", True),
    ("grpc.default_compression_algorithm", grpc.Compression.Gzip.value),
]


class EmbeddingClient:
    """
    gRPC client for the embedding service.

    Example usage:
        client = EmbeddingClient("192.168.2.1:50051")  # Thunderbolt IP
        embedding = client.embed("Hei, verden!")
        similarity = client.similarity("Oslo", "Hovedstaden i Norge")
    """

    def __init__(self, address: str = "localhost:50051"):
        """
        Initialize the client.

        Args:
            address: Server address (host:port)
        """
        if embedding_pb2 is None:
            raise ImportError(
                "gRPC generated files not found. Run 'scripts/generate_grpc.sh' first."
            )

        self.address = address
        self.channel = grpc.insecure_channel(address, options=GRPC_OPTIONS)
        self.stub = embedding_pb2_grpc.EmbeddingServiceStub(self.channel)
        logger.debug(f"Connected to {address}")

    def close(self):
        """Close the channel."""
        self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def embed(
        self,
        text: str,
        normalize: bool = True,
    ) -> Tuple[List[float], int]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            normalize: Whether to L2-normalize

        Returns:
            Tuple of (embedding vector, dimension)
        """
        request = embedding_pb2.EmbeddingRequest(
            text=text,
            mode=embedding_pb2.MODE_EMBEDDING,
            output=embedding_pb2.OUTPUT_VECTOR,
            normalize=normalize,
        )
        response = self.stub.Embed(request)
        return list(response.vector.values), response.vector.dimension

    def embed_batch(
        self,
        texts: List[str],
        normalize: bool = True,
    ) -> Tuple[List[List[float]], int]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            normalize: Whether to L2-normalize

        Returns:
            Tuple of (list of embedding vectors, dimension)
        """
        request = embedding_pb2.BatchEmbeddingRequest(
            texts=texts,
            mode=embedding_pb2.MODE_EMBEDDING,
            output=embedding_pb2.OUTPUT_VECTOR,
            normalize=normalize,
        )
        response = self.stub.EmbedBatch(request)
        vectors = [list(v.values) for v in response.vectors]
        return vectors, response.vectors[0].dimension if vectors else 768

    def embed_stream(
        self,
        chunks: List[str],
        normalize: bool = True,
    ) -> Iterator[Tuple[int, List[float]]]:
        """
        Stream embeddings for multiple text chunks.

        Args:
            chunks: List of text chunks
            normalize: Whether to L2-normalize

        Yields:
            Tuples of (chunk_index, embedding vector)
        """

        def request_generator():
            # Send config first
            yield embedding_pb2.StreamingEmbeddingRequest(
                config=embedding_pb2.StreamConfig(
                    mode=embedding_pb2.MODE_EMBEDDING,
                    output=embedding_pb2.OUTPUT_VECTOR,
                    normalize=normalize,
                )
            )
            # Send chunks
            for i, chunk in enumerate(chunks):
                yield embedding_pb2.StreamingEmbeddingRequest(
                    chunk=embedding_pb2.TextChunk(
                        text=chunk,
                        chunk_index=i,
                        is_last=(i == len(chunks) - 1),
                    )
                )

        for response in self.stub.EmbedStream(request_generator()):
            yield response.chunk_index, list(response.vector.values)

    def encode(
        self,
        text: str,
        include_tokens: bool = False,
        normalize: bool = True,
    ) -> dict:
        """
        Generate NB-BERT encoder representation.

        Args:
            text: Text to encode
            include_tokens: Whether to include token embeddings
            normalize: Whether to L2-normalize

        Returns:
            Dictionary with pooled_embedding, token_embeddings, tokens
        """
        request = embedding_pb2.EmbeddingRequest(
            text=text,
            mode=embedding_pb2.MODE_ENCODE,
            output=(
                embedding_pb2.OUTPUT_TOKEN_VECTORS
                if include_tokens
                else embedding_pb2.OUTPUT_POOLED_VECTOR
            ),
            normalize=normalize,
        )
        response = self.stub.Encode(request)

        result = {
            "pooled_embedding": list(response.pooled_embedding.values),
            "dimension": response.pooled_embedding.dimension,
            "model": response.model,
        }

        if response.token_embeddings:
            result["token_embeddings"] = [
                list(v.values) for v in response.token_embeddings
            ]

        if response.tokens:
            result["tokens"] = list(response.tokens)

        return result

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (-1 to 1)
        """
        request = embedding_pb2.SimilarityRequest(text1=text1, text2=text2)
        response = self.stub.Similarity(request)
        return response.similarity

    def bulk_similarity(
        self,
        query: str,
        documents: List[str],
        top_k: int = 0,
    ) -> List[dict]:
        """
        Calculate similarity between query and multiple documents.

        Args:
            query: Query text
            documents: List of document texts
            top_k: Return top K results (0 = all)

        Returns:
            List of dicts with index, similarity, document
        """
        request = embedding_pb2.BulkSimilarityRequest(
            query=query,
            documents=documents,
            top_k=top_k,
        )
        response = self.stub.BulkSimilarity(request)
        return [
            {
                "index": r.index,
                "similarity": r.similarity,
                "document": r.document,
            }
            for r in response.results
        ]

    def health(self) -> dict:
        """Check server health."""
        request = embedding_pb2.HealthRequest()
        response = self.stub.Health(request)
        return {
            "status": response.status,
            "sbert_model_loaded": response.sbert_model_loaded,
            "bert_model_loaded": response.bert_model_loaded,
            "device": response.device,
            "version": response.version,
        }


# Convenience functions for one-off calls
def embed(text: str, address: str = "localhost:50051") -> List[float]:
    """Quick embed a single text."""
    with EmbeddingClient(address) as client:
        vector, _ = client.embed(text)
        return vector


def similarity(text1: str, text2: str, address: str = "localhost:50051") -> float:
    """Quick similarity calculation."""
    with EmbeddingClient(address) as client:
        return client.similarity(text1, text2)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Test the gRPC embedding client")
    parser.add_argument("--address", default="localhost:50051", help="Server address")
    parser.add_argument("--text", default="Hei, verden!", help="Text to embed")
    args = parser.parse_args()

    client = EmbeddingClient(args.address)

    print("Health check:")
    print(client.health())

    print(f"\nEmbedding for: {args.text}")
    vec, dim = client.embed(args.text)
    print(f"Dimension: {dim}")
    print(f"First 10 values: {vec[:10]}")

    client.close()
