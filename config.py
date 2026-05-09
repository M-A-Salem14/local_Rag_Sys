# Import the Path class from the pathlib module.
# pathlib is a standard library module in Python that provides an object-oriented interface for filesystem paths.
# The Path class represents filesystem paths and provides methods for path manipulation, making it easier and more robust than using strings for paths.
# This import is necessary for defining and working with file and directory paths in the configuration.
import sys
import torch
from pathlib import Path


def _resolve_device() -> str:
    """Probe CUDA and return 'cuda' if working, 'cpu' with a warning if not."""
    if not torch.cuda.is_available():
        print(
            "\033[91m[WARNING] CUDA not available — running on CPU. "
            "This will be significantly slower.\033[0m",
            file=sys.stderr,
        )
        return "cpu"
    try:
        _ = torch.zeros(1, device="cuda")
        del _
        torch.cuda.empty_cache()
        return "cuda"
    except RuntimeError as e:
        print(
            f"\033[91m[WARNING] CUDA probe failed ({e}) — "
            f"falling back to CPU. This will be significantly slower.\033[0m",
            file=sys.stderr,
        )
        return "cpu"


DEVICE = _resolve_device()

# ============================================================================
# PATHS AND DIRECTORIES
# ============================================================================
# Set the base directory to the current directory (where this config.py is located).
# Using Path(".") represents the root directory of the project, making all other paths relative to this point.
BASE_DIR     = Path(".")

# Define the data directory where raw input documents are stored.
# This folder should contain the source files (PDF, TXT, etc.) that will be indexed into the vector database.
# The / operator from pathlib joins paths in a cross-platform compatible way (works on Windows, Linux, macOS).
DATA_DIR     = BASE_DIR / "data"

# Define the database directory where LanceDB vector store files will be persisted.
# LanceDB automatically creates this directory and stores indexed vectors and metadata here.
# This is the local persistent storage for the knowledge base after indexing.
DB_DIR       = BASE_DIR / "db"

# Define the models directory for caching downloaded ONNX and embedding models.
# Pre-trained models are stored locally here to avoid re-downloading on subsequent runs.
# This improves startup time and allows offline operation after initial download.
MODELS_DIR   = BASE_DIR / "models"

# ============================================================================
# CHUNKING DEBUG LOG
# ============================================================================
# Output directory for chunking debug reports.
# Reports are timestamped Markdown files written here after each index run.
CHUNKING_LOG_DIR  = BASE_DIR / "logs"

# Logging mode controls how much detail is captured:
#   "metadata" — file summaries, chunk word/char counts, overlap checks, anomalies
#   "full"     — everything in metadata PLUS full chunk text content and overlap regions
CHUNKING_LOG_MODE = "metadata"

# ============================================================================
# CHUNKING STRATEGY
# ============================================================================
# Select chunking algorithm:
#   "legacy"   — original word-based sliding window (512 words, 64 overlap)
#   "markdown" — hybrid: split on markdown headers → paragraphs → sentences
CHUNKING_STRATEGY  = "markdown"

# Parameters for "markdown" strategy:
CHUNK_MAX_WORDS    = 512   # max words per chunk before sub-splitting
CHUNK_MIN_WORDS    = 50    # sections below this get merged with the next
CHUNK_OVERLAP_SENT = 1     # sentence overlap between sub-split chunks
# ============================================================================
# Specify the name of the table in LanceDB where documents will be stored.
# This table holds the embeddings, document chunks, and metadata for all indexed documents.
# The table structure is automatically created during the first indexing operation.
TABLE_NAME   = "documents"

# ============================================================================
# EMBEDDING MODEL CONFIGURATION (BGE-M3)
# ============================================================================
# Specify the embedding model to use for converting text into dense vectors.
# BGE-M3 (Base General Embedding Model 3) is a multilingual embedding model from BAAI.
# It supports both dense embeddings and sparse embeddings, enabling hybrid search capabilities.
EMBED_MODEL  = "BAAI/bge-m3"

# Specify the dimensionality of the embeddings produced by the BGE-M3 model.
# BGE-M3 produces 1024-dimensional vectors, which is a good balance between accuracy and computational cost.
# This dimension is used when creating the LanceDB table schema.
EMBED_DIM    = 1024

# ============================================================================
# RERANKER MODEL CONFIGURATION
# ============================================================================
# Specify the cross-encoder reranker model for refining search results.
# BGE Reranker v2-M3 is a fine-tuned model that re-scores retrieved documents by computing relevance scores.
# Reranking improves the quality of results by identifying the most relevant documents from initial retrieval.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Set the number of documents to rerank and send to the LLM for generation.
# This is the final set of high-quality documents after reranking, reducing context and improving relevance.
# Smaller values (5-10) reduce latency and token usage; larger values may improve answer completeness.
RERANK_TOP_K   = 5

# Set the number of candidate documents to retrieve before reranking.
# Initial retrieval fetches more documents than needed, then the reranker selects the best ones.
# Larger values (10-50) increase the chance of finding relevant documents but require more computation.
RETRIEVE_TOP_K = 20

# ============================================================================
# OLLAMA LLM CONFIGURATION
# ============================================================================
# Specify the base URL for the Ollama API server.
# Ollama runs as a local service, typically on localhost (127.0.0.1) with default port 11434.
# This endpoint is used to send prompts and generation requests to the locally running LLM.
OLLAMA_BASE  = "http://localhost:11434"

# Specify the general-purpose language model for standard RAG queries.
# Qwen3:14b is a 14-billion parameter model suitable for general text generation and reasoning tasks.
# Different models can be used depending on the task requirements and available compute resources.
LLM_GENERAL  = "qwen3:14b"

# Specify the specialized model for complex reasoning and logical inference tasks.
# Deepseek-R1:14b is optimized for reasoning-heavy tasks and multi-step problem solving.
# Using different models allows routing queries to the most appropriate model for the task.
# LLM_REASON   = "deepseek-r1:14b"
LLM_REASON   = "qwen3:14b" #TEMP: Use general model for all queries until we have DeepSeek ready locally ---

# ============================================================================
# HYBRID RETRIEVAL WEIGHTS (RRF Fusion)
# ============================================================================
# Set the weight for dense vector similarity in hybrid search scoring.
# Dense embeddings capture semantic meaning and are generally effective for semantic search.
# A weight of 0.6 gives 60% importance to dense similarity, balancing with sparse retrieval.
DENSE_WEIGHT  = 0.6

# Set the weight for sparse (keyword) retrieval in hybrid search scoring.
# Sparse embeddings are effective for exact keyword matching and technical terms.
# A weight of 0.4 gives 40% importance to keyword matching, complementing dense retrieval.
# Together, DENSE_WEIGHT + SPARSE_WEIGHT should equal 1.0 for normalized scoring.
SPARSE_WEIGHT = 0.4