"""
gRPC service implementation for NB-BERT/NB-SBERT embeddings.

Optimized for high-bandwidth connections like Thunderbolt 5.
"""

from typing import Iterator

import grpc
import numpy as np
from loguru import logger

from app import __version__
from app.config import Settings
from app.services.embedding_service import EmbeddingService, get_embedding_service

# Import generated protobuf classes
# These will be available after running scripts/generate_grpc.sh
try:
    from app.grpc.generated import embedding_pb2
    from app.grpc.generated import embedding_pb2_grpc
except ImportError:
    embedding_pb2 = None
    embedding_pb2_grpc = None


class EmbeddingGRPCService:
    """
    gRPC service implementation for embedding generation.

    Provides high-performance binary RPC for:
    - Single text embedding
    - Batch embedding
    - Streaming embedding for large documents
    - Similarity calculation
    """

    def __init__(self, settings: Settings):
        """Initialize the gRPC service."""
        self.settings = settings
        self._embedding_service: EmbeddingService = None

    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy-load the embedding service."""
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service(self.settings)
        return self._embedding_service

    def _create_vector(self, values: list, dimension: int = 768) -> "embedding_pb2.Vector":
        """Create a Vector protobuf message."""
        return embedding_pb2.Vector(values=values, dimension=dimension)

    def Embed(
        self,
        request: "embedding_pb2.EmbeddingRequest",
        context: grpc.ServicerContext,
    ) -> "embedding_pb2.EmbeddingResponse":
        """Generate embedding for a single text."""
        try:
            vector, dimension = self.embedding_service.embed(
                text=request.text,
                normalize=request.normalize,
            )

            return embedding_pb2.EmbeddingResponse(
                vector=self._create_vector(vector, dimension),
                model=self.settings.sbert_model_name,
                normalized=request.normalize,
            )

        except Exception as e:
            logger.error(f"Embed failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return embedding_pb2.EmbeddingResponse()

    def EmbedBatch(
        self,
        request: "embedding_pb2.BatchEmbeddingRequest",
        context: grpc.ServicerContext,
    ) -> "embedding_pb2.BatchEmbeddingResponse":
        """Generate embeddings for multiple texts."""
        try:
            texts = list(request.texts)

            if len(texts) > self.settings.max_batch_size:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    f"Batch size {len(texts)} exceeds maximum {self.settings.max_batch_size}"
                )
                return embedding_pb2.BatchEmbeddingResponse()

            vectors, dimension = self.embedding_service.embed_batch(
                texts=texts,
                normalize=request.normalize,
            )

            proto_vectors = [
                self._create_vector(v, dimension) for v in vectors
            ]

            return embedding_pb2.BatchEmbeddingResponse(
                vectors=proto_vectors,
                count=len(vectors),
                model=self.settings.sbert_model_name,
                normalized=request.normalize,
            )

        except Exception as e:
            logger.error(f"EmbedBatch failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return embedding_pb2.BatchEmbeddingResponse()

    def EmbedStream(
        self,
        request_iterator: Iterator["embedding_pb2.StreamingEmbeddingRequest"],
        context: grpc.ServicerContext,
    ) -> Iterator["embedding_pb2.StreamingEmbeddingResponse"]:
        """
        Stream embeddings for large documents.

        Client sends text chunks, server responds with embeddings for each chunk.
        Useful for processing large documents over high-bandwidth connections.
        """
        config = None
        normalize = True

        try:
            for request in request_iterator:
                # Handle config message
                if request.HasField("config"):
                    config = request.config
                    normalize = config.normalize
                    continue

                # Handle text chunk
                if request.HasField("chunk"):
                    chunk = request.chunk
                    vector, dimension = self.embedding_service.embed(
                        text=chunk.text,
                        normalize=normalize,
                    )

                    yield embedding_pb2.StreamingEmbeddingResponse(
                        chunk_index=chunk.chunk_index,
                        vector=self._create_vector(vector, dimension),
                        is_last=chunk.is_last,
                    )

        except Exception as e:
            logger.error(f"EmbedStream failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def Encode(
        self,
        request: "embedding_pb2.EmbeddingRequest",
        context: grpc.ServicerContext,
    ) -> "embedding_pb2.EncoderResponse":
        """Generate NB-BERT encoder representation."""
        try:
            # Determine output type
            include_tokens = (
                request.output == embedding_pb2.OUTPUT_TOKEN_VECTORS
            )

            result = self.embedding_service.encode(
                text=request.text,
                output_type="token_vectors" if include_tokens else "pooled_vector",
                normalize=request.normalize,
                return_tokens=include_tokens,
            )

            response = embedding_pb2.EncoderResponse(
                pooled_embedding=self._create_vector(
                    result["pooled_embedding"],
                    result["dimension"],
                ),
                model=self.settings.bert_model_name,
                normalized=request.normalize,
            )

            # Add token embeddings if requested
            if result.get("token_embeddings"):
                for token_vec in result["token_embeddings"]:
                    response.token_embeddings.append(
                        self._create_vector(token_vec, result["dimension"])
                    )

            # Add tokens if available
            if result.get("tokens"):
                response.tokens.extend(result["tokens"])

            return response

        except Exception as e:
            logger.error(f"Encode failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return embedding_pb2.EncoderResponse()

    def Similarity(
        self,
        request: "embedding_pb2.SimilarityRequest",
        context: grpc.ServicerContext,
    ) -> "embedding_pb2.SimilarityResponse":
        """Calculate cosine similarity between two texts."""
        try:
            vec1, _ = self.embedding_service.embed(request.text1, normalize=True)
            vec2, _ = self.embedding_service.embed(request.text2, normalize=True)

            # Cosine similarity (dot product of normalized vectors)
            similarity = float(np.dot(vec1, vec2))

            return embedding_pb2.SimilarityResponse(
                similarity=similarity,
                model=self.settings.sbert_model_name,
            )

        except Exception as e:
            logger.error(f"Similarity failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return embedding_pb2.SimilarityResponse()

    def BulkSimilarity(
        self,
        request: "embedding_pb2.BulkSimilarityRequest",
        context: grpc.ServicerContext,
    ) -> "embedding_pb2.BulkSimilarityResponse":
        """
        Calculate similarity between a query and multiple documents.

        Returns results sorted by similarity (highest first).
        """
        try:
            documents = list(request.documents)

            # Embed query
            query_vec, _ = self.embedding_service.embed(request.query, normalize=True)
            query_vec = np.array(query_vec)

            # Embed all documents in batch
            doc_vecs, _ = self.embedding_service.embed_batch(documents, normalize=True)
            doc_vecs = np.array(doc_vecs)

            # Calculate similarities
            similarities = np.dot(doc_vecs, query_vec)

            # Sort by similarity
            sorted_indices = np.argsort(similarities)[::-1]

            # Apply top_k limit if specified
            if request.top_k > 0:
                sorted_indices = sorted_indices[: request.top_k]

            # Build results
            results = []
            for idx in sorted_indices:
                results.append(
                    embedding_pb2.SimilarityResult(
                        index=int(idx),
                        similarity=float(similarities[idx]),
                        document=documents[idx],
                    )
                )

            return embedding_pb2.BulkSimilarityResponse(
                results=results,
                model=self.settings.sbert_model_name,
            )

        except Exception as e:
            logger.error(f"BulkSimilarity failed: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return embedding_pb2.BulkSimilarityResponse()

    def Health(
        self,
        request: "embedding_pb2.HealthRequest",
        context: grpc.ServicerContext,
    ) -> "embedding_pb2.HealthResponse":
        """Health check endpoint."""
        return embedding_pb2.HealthResponse(
            status="healthy",
            sbert_model_loaded=self.embedding_service.sbert_loaded,
            bert_model_loaded=self.embedding_service.bert_loaded,
            device=self.embedding_service.device,
            version=__version__,
        )


def create_servicer(settings: Settings) -> "embedding_pb2_grpc.EmbeddingServiceServicer":
    """Create a gRPC servicer instance."""
    if embedding_pb2_grpc is None:
        raise ImportError(
            "gRPC generated files not found. Run 'scripts/generate_grpc.sh' first."
        )

    class Servicer(embedding_pb2_grpc.EmbeddingServiceServicer):
        """gRPC servicer wrapping EmbeddingGRPCService."""

        def __init__(self):
            self.service = EmbeddingGRPCService(settings)

        def Embed(self, request, context):
            return self.service.Embed(request, context)

        def EmbedBatch(self, request, context):
            return self.service.EmbedBatch(request, context)

        def EmbedStream(self, request_iterator, context):
            return self.service.EmbedStream(request_iterator, context)

        def Encode(self, request, context):
            return self.service.Encode(request, context)

        def Similarity(self, request, context):
            return self.service.Similarity(request, context)

        def BulkSimilarity(self, request, context):
            return self.service.BulkSimilarity(request, context)

        def Health(self, request, context):
            return self.service.Health(request, context)

    return Servicer()
