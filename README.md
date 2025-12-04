# llmServe - Norwegian Embedding Generator

A high-performance embedding service for generating embeddings using **NB-BERT** and **NB-SBERT** models from the National Library of Norway (NB AiLab).

**Dual Protocol Support:**
- **REST API** (FastAPI) - Standard HTTP/JSON interface
- **gRPC** - High-performance binary protocol optimized for Thunderbolt 5

Optimized for:
- M series cpus
- Thunderbolt 5 connections between machines (~80 Gbps)
- RAG (Retrieval-Augmented Generation) pipelines
- Norwegian text (bokmål, nynorsk, dialects, historical texts)

## Features

- **NB-SBERT Embeddings**: 768-dimensional sentence embeddings for semantic similarity
- **NB-BERT Encoder**: Token-level encoder representations
- **Dual Protocol**: REST API + gRPC for flexibility and performance
- **Batch Processing**: Efficient batch embedding generation
- **Streaming**: gRPC streaming for large documents
- **Bulk Similarity**: Query vs many documents in single call
- **Text Preprocessing**: Automatic noise removal (URLs, emojis, courtesy phrases)
- **Deterministic Output**: Consistent embeddings across requests
- **L2 Normalization**: Optional vector normalization for cosine similarity

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd llmServe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and configure as needed:

```bash
cp .env.example .env
```

Key configuration options:

| Variable | Default | Description |
|----------|---------|-------------|
| `SBERT_MODEL_NAME` | `NbAiLab/nb-sbert-base` | Sentence embedding model |
| `BERT_MODEL_NAME` | `NbAiLab/nb-bert-base` | Encoder model |
| `DEVICE` | `cpu` | Device: `cpu`, `cuda`, or `mps` |
| `MAX_BATCH_SIZE` | `32` | Maximum texts per batch |
| `PORT` | `8000` | Server port |

### Generate gRPC Code

Before using gRPC, generate the Python code from protobuf definitions:

```bash
# Make script executable (first time only)
chmod +x scripts/generate_grpc.sh

# Generate gRPC code
./scripts/generate_grpc.sh
```

### Running the Service

**REST API (FastAPI):**
```bash
# Production
python run.py

# Development with auto-reload
python run.py --reload

# Custom host/port
python run.py --host 0.0.0.0 --port 8080
```

**gRPC Server (Thunderbolt optimized):**
```bash
# Standard gRPC server
python run_grpc.py

# Auto-detect Thunderbolt interface
python run_grpc.py --thunderbolt

# Custom settings
python run_grpc.py --host 192.168.2.1 --port 50051 --workers 20

# Preload models for faster first request
python run_grpc.py --preload
```

The services will be available at:
- REST API: `http://localhost:8000`
- gRPC: `localhost:50051`

## API Endpoints

### Generate Embedding

**POST** `/api/v1/embeddings`

Generate embedding for a single text.

**Request:**
```json
{
  "text": "Sykehjemmet tilbyr korttidsopphold, langtidsopphold og dagtilbud.",
  "mode": "embedding",
  "output": "vector",
  "normalize": true
}
```

**Response:**
```json
{
  "vector": [0.0234, -0.0156, ...],
  "dimension": 768,
  "model": "NbAiLab/nb-sbert-base",
  "normalized": true
}
```

### Batch Embeddings

**POST** `/api/v1/embeddings/batch`

Generate embeddings for multiple texts.

**Request:**
```json
{
  "texts": [
    "Første setning",
    "Andre setning",
    "Tredje setning"
  ],
  "mode": "embedding",
  "normalize": true
}
```

**Response:**
```json
{
  "vectors": [[...], [...], [...]],
  "dimension": 768,
  "count": 3,
  "model": "NbAiLab/nb-sbert-base",
  "normalized": true
}
```

### NB-BERT Encoder

**POST** `/api/v1/embeddings`

Use `mode: "encode"` for NB-BERT encoder output with token-level embeddings.

**Request:**
```json
{
  "text": "Eksempel på norsk tekst",
  "mode": "encode",
  "output": "token_vectors",
  "normalize": true
}
```

**Response:**
```json
{
  "pooled_embedding": [...],
  "token_embeddings": [[...], [...], ...],
  "tokens": ["[CLS]", "eksempel", "på", "norsk", "tekst", "[SEP]"],
  "dimension": 768,
  "model": "NbAiLab/nb-bert-base",
  "normalized": true
}
```

### Calculate Similarity

**POST** `/api/v1/embeddings/similarity?text1=...&text2=...`

Calculate cosine similarity between two texts.

**Response:**
```json
{
  "text1": "Oslo er hovedstaden i Norge",
  "text2": "Bergen er en by på vestlandet",
  "similarity": 0.6234,
  "model": "NbAiLab/nb-sbert-base"
}
```

### Health Check

**GET** `/api/v1/health`

```json
{
  "status": "healthy",
  "sbert_model_loaded": true,
  "bert_model_loaded": false,
  "device": "cpu",
  "version": "0.1.0"
}
```

## gRPC API (Thunderbolt Optimized)

The gRPC interface provides higher throughput and lower latency than REST,
especially over high-bandwidth connections like Thunderbolt 5.

### Setup Thunderbolt Networking

1. Connect two MacBooks via Thunderbolt cable
2. Configure IP addresses:
   - MacBook 1 (server): `192.168.2.1`
   - MacBook 2 (client): `192.168.2.2`
3. Start the gRPC server on MacBook 1:
   ```bash
   python run_grpc.py --host 192.168.2.1
   ```

### Python Client Usage

```python
from app.grpc.client import EmbeddingClient

# Connect to server over Thunderbolt
client = EmbeddingClient("192.168.2.1:50051")

# Single embedding
vector, dim = client.embed("Hei, verden!")
print(f"Dimension: {dim}, First 5 values: {vector[:5]}")

# Batch embedding
vectors, dim = client.embed_batch([
    "Første setning",
    "Andre setning",
    "Tredje setning",
])

# Similarity
sim = client.similarity(
    "Oslo er hovedstaden",
    "Hovedstaden i Norge heter Oslo"
)
print(f"Similarity: {sim}")

# Bulk similarity (query vs many documents)
results = client.bulk_similarity(
    query="Hva er hovedstaden i Norge?",
    documents=[
        "Oslo er hovedstaden i Norge",
        "Bergen ligger på vestlandet",
        "Trondheim har NTNU",
    ],
    top_k=2,
)
for r in results:
    print(f"{r['similarity']:.3f}: {r['document']}")

# Streaming for large documents
chunks = ["Første del av dokumentet...", "Andre del...", "Siste del..."]
for chunk_idx, embedding in client.embed_stream(chunks):
    print(f"Chunk {chunk_idx}: {len(embedding)} dims")

# Health check
print(client.health())

client.close()
```

### gRPC Methods

| Method | Description |
|--------|-------------|
| `Embed` | Single text embedding |
| `EmbedBatch` | Multiple texts in one call |
| `EmbedStream` | Streaming for large documents |
| `Encode` | NB-BERT token-level representations |
| `Similarity` | Cosine similarity between two texts |
| `BulkSimilarity` | Query vs many documents |
| `Health` | Service health check |

### Performance Comparison

| Protocol | Latency (1 text) | Throughput (batch 32) |
|----------|------------------|----------------------|
| REST/JSON | ~5-10ms | ~50-100ms |
| gRPC/Protobuf | ~1-3ms | ~20-40ms |

*Measurements on Thunderbolt 5 connection, models preloaded*

## Modes and Output Types

### Modes

| Mode | Model | Use Case |
|------|-------|----------|
| `embedding` | NB-SBERT | Sentence embeddings for RAG/search |
| `encode` | NB-BERT | Token-level representations |

### Output Types

| Output | Description |
|--------|-------------|
| `vector` | Single 768-dim pooled vector |
| `token_vectors` | Per-token embeddings (with mode=encode) |
| `pooled_vector` | Pooled representation from BERT |

## Text Preprocessing

The service automatically cleans input text:

**Removed:**
- Courtesy phrases (vennligst, mvh, med vennlig hilsen)
- Contact information (phone, email, URLs)
- Emojis and special characters

**Preserved:**
- Semantic content (entities, actions, relationships)
- Norwegian characters (æ, ø, å)

## Vector Database Integration

The embeddings are compatible with:

- **ArangoDB** (with vector index)
- **OpenSearch** (with kNN plugin)
- **Weaviate**
- **Redis Vector** (RediSearch)
- **Pinecone**
- **ChromaDB**
- **FAISS**

Example: Storing in ChromaDB

```python
import chromadb
import requests

# Generate embedding
response = requests.post(
    "http://localhost:8000/api/v1/embeddings",
    json={"text": "Eksempel tekst", "normalize": True}
)
embedding = response.json()["vector"]

# Store in ChromaDB
client = chromadb.Client()
collection = client.create_collection("documents")
collection.add(
    embeddings=[embedding],
    documents=["Eksempel tekst"],
    ids=["doc1"]
)
```

## Development

### Project Structure

```
llmServe/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── api/                 # REST API
│   │   ├── router.py
│   │   └── v1/
│   │       ├── embeddings.py
│   │       └── health.py
│   ├── grpc/                # gRPC server
│   │   ├── server.py        # gRPC server
│   │   ├── service.py       # Service implementation
│   │   ├── client.py        # Python client
│   │   └── generated/       # Generated protobuf code
│   ├── models/
│   │   └── embedding.py     # Pydantic models
│   └── services/
│       └── embedding_service.py  # Shared embedding logic
├── proto/
│   └── embedding.proto      # Protobuf definitions
├── scripts/
│   └── generate_grpc.sh     # Code generation script
├── tests/
├── requirements.txt
├── run.py                   # REST API entry point
├── run_grpc.py              # gRPC server entry point
└── README.md
```

### Running Tests

```bash
pytest tests/ -v
```

## Models

### NB-SBERT (nb-sbert-base)

- **Architecture**: Sentence-BERT based on NB-BERT
- **Dimensions**: 768
- **Training**: Norwegian text similarity tasks
- **Best for**: Semantic search, document similarity, clustering

### NB-BERT (nb-bert-base)

- **Architecture**: BERT base (12 layers, 768 hidden, 12 heads)
- **Dimensions**: 768 per token
- **Training**: Norwegian text from various sources
- **Best for**: Token-level tasks, NER, classification

## Performance Notes

- First request will load models (may take 10-30 seconds)
- Subsequent requests are fast (~50-200ms depending on text length)
- Batch processing is more efficient than individual requests
- CPU mode works well on Apple Silicon (M1/M2)

## License

See the LICENSE file for details.

## Acknowledgements

Models provided by [NB AiLab](https://github.com/NbAiLab) (National Library of Norway AI Lab).
