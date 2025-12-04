from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    MODE_UNSPECIFIED: _ClassVar[Mode]
    MODE_EMBEDDING: _ClassVar[Mode]
    MODE_ENCODE: _ClassVar[Mode]

class OutputType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OUTPUT_UNSPECIFIED: _ClassVar[OutputType]
    OUTPUT_VECTOR: _ClassVar[OutputType]
    OUTPUT_TOKEN_VECTORS: _ClassVar[OutputType]
    OUTPUT_POOLED_VECTOR: _ClassVar[OutputType]
MODE_UNSPECIFIED: Mode
MODE_EMBEDDING: Mode
MODE_ENCODE: Mode
OUTPUT_UNSPECIFIED: OutputType
OUTPUT_VECTOR: OutputType
OUTPUT_TOKEN_VECTORS: OutputType
OUTPUT_POOLED_VECTOR: OutputType

class EmbeddingRequest(_message.Message):
    __slots__ = ("text", "mode", "output", "normalize")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_FIELD_NUMBER: _ClassVar[int]
    text: str
    mode: Mode
    output: OutputType
    normalize: bool
    def __init__(self, text: _Optional[str] = ..., mode: _Optional[_Union[Mode, str]] = ..., output: _Optional[_Union[OutputType, str]] = ..., normalize: bool = ...) -> None: ...

class BatchEmbeddingRequest(_message.Message):
    __slots__ = ("texts", "mode", "output", "normalize")
    TEXTS_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_FIELD_NUMBER: _ClassVar[int]
    texts: _containers.RepeatedScalarFieldContainer[str]
    mode: Mode
    output: OutputType
    normalize: bool
    def __init__(self, texts: _Optional[_Iterable[str]] = ..., mode: _Optional[_Union[Mode, str]] = ..., output: _Optional[_Union[OutputType, str]] = ..., normalize: bool = ...) -> None: ...

class StreamingEmbeddingRequest(_message.Message):
    __slots__ = ("config", "chunk")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    config: StreamConfig
    chunk: TextChunk
    def __init__(self, config: _Optional[_Union[StreamConfig, _Mapping]] = ..., chunk: _Optional[_Union[TextChunk, _Mapping]] = ...) -> None: ...

class StreamConfig(_message.Message):
    __slots__ = ("mode", "output", "normalize", "chunk_size")
    MODE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_FIELD_NUMBER: _ClassVar[int]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    mode: Mode
    output: OutputType
    normalize: bool
    chunk_size: int
    def __init__(self, mode: _Optional[_Union[Mode, str]] = ..., output: _Optional[_Union[OutputType, str]] = ..., normalize: bool = ..., chunk_size: _Optional[int] = ...) -> None: ...

class TextChunk(_message.Message):
    __slots__ = ("text", "chunk_index", "is_last")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    text: str
    chunk_index: int
    is_last: bool
    def __init__(self, text: _Optional[str] = ..., chunk_index: _Optional[int] = ..., is_last: bool = ...) -> None: ...

class Vector(_message.Message):
    __slots__ = ("values", "dimension")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    dimension: int
    def __init__(self, values: _Optional[_Iterable[float]] = ..., dimension: _Optional[int] = ...) -> None: ...

class EmbeddingResponse(_message.Message):
    __slots__ = ("vector", "model", "normalized")
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    NORMALIZED_FIELD_NUMBER: _ClassVar[int]
    vector: Vector
    model: str
    normalized: bool
    def __init__(self, vector: _Optional[_Union[Vector, _Mapping]] = ..., model: _Optional[str] = ..., normalized: bool = ...) -> None: ...

class EncoderResponse(_message.Message):
    __slots__ = ("pooled_embedding", "token_embeddings", "tokens", "model", "normalized")
    POOLED_EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    TOKEN_EMBEDDINGS_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    NORMALIZED_FIELD_NUMBER: _ClassVar[int]
    pooled_embedding: Vector
    token_embeddings: _containers.RepeatedCompositeFieldContainer[Vector]
    tokens: _containers.RepeatedScalarFieldContainer[str]
    model: str
    normalized: bool
    def __init__(self, pooled_embedding: _Optional[_Union[Vector, _Mapping]] = ..., token_embeddings: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ..., tokens: _Optional[_Iterable[str]] = ..., model: _Optional[str] = ..., normalized: bool = ...) -> None: ...

class BatchEmbeddingResponse(_message.Message):
    __slots__ = ("vectors", "count", "model", "normalized")
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    NORMALIZED_FIELD_NUMBER: _ClassVar[int]
    vectors: _containers.RepeatedCompositeFieldContainer[Vector]
    count: int
    model: str
    normalized: bool
    def __init__(self, vectors: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ..., count: _Optional[int] = ..., model: _Optional[str] = ..., normalized: bool = ...) -> None: ...

class StreamingEmbeddingResponse(_message.Message):
    __slots__ = ("chunk_index", "vector", "is_last")
    CHUNK_INDEX_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    chunk_index: int
    vector: Vector
    is_last: bool
    def __init__(self, chunk_index: _Optional[int] = ..., vector: _Optional[_Union[Vector, _Mapping]] = ..., is_last: bool = ...) -> None: ...

class SimilarityRequest(_message.Message):
    __slots__ = ("text1", "text2")
    TEXT1_FIELD_NUMBER: _ClassVar[int]
    TEXT2_FIELD_NUMBER: _ClassVar[int]
    text1: str
    text2: str
    def __init__(self, text1: _Optional[str] = ..., text2: _Optional[str] = ...) -> None: ...

class SimilarityResponse(_message.Message):
    __slots__ = ("similarity", "model")
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    similarity: float
    model: str
    def __init__(self, similarity: _Optional[float] = ..., model: _Optional[str] = ...) -> None: ...

class BulkSimilarityRequest(_message.Message):
    __slots__ = ("query", "documents", "top_k")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    query: str
    documents: _containers.RepeatedScalarFieldContainer[str]
    top_k: int
    def __init__(self, query: _Optional[str] = ..., documents: _Optional[_Iterable[str]] = ..., top_k: _Optional[int] = ...) -> None: ...

class SimilarityResult(_message.Message):
    __slots__ = ("index", "similarity", "document")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    DOCUMENT_FIELD_NUMBER: _ClassVar[int]
    index: int
    similarity: float
    document: str
    def __init__(self, index: _Optional[int] = ..., similarity: _Optional[float] = ..., document: _Optional[str] = ...) -> None: ...

class BulkSimilarityResponse(_message.Message):
    __slots__ = ("results", "model")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[SimilarityResult]
    model: str
    def __init__(self, results: _Optional[_Iterable[_Union[SimilarityResult, _Mapping]]] = ..., model: _Optional[str] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status", "sbert_model_loaded", "bert_model_loaded", "device", "version")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    SBERT_MODEL_LOADED_FIELD_NUMBER: _ClassVar[int]
    BERT_MODEL_LOADED_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    status: str
    sbert_model_loaded: bool
    bert_model_loaded: bool
    device: str
    version: str
    def __init__(self, status: _Optional[str] = ..., sbert_model_loaded: bool = ..., bert_model_loaded: bool = ..., device: _Optional[str] = ..., version: _Optional[str] = ...) -> None: ...
