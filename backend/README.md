# Semantic Fashion Image Search API

A production-ready FastAPI backend for semantic fashion search using MongoDB Atlas Vector Search and HuggingFace CLIP embeddings.

## 🎯 What This API Does

This API enables semantic search for fashion products using state-of-the-art AI technology:

- **Text-to-Image Search**: Find fashion items by natural language description ("red floral summer dress")
- **Image-to-Image Search**: Find visually similar products by uploading a reference image
- **Cross-Modal Understanding**: Uses HuggingFace CLIP (`openai/clip-vit-large-patch14`) to understand both text and visual content in the same semantic space
- **High Performance**: Optimized for large-scale fashion catalogs with vector indexing

## 🧠 Learning Concepts Demonstrated

### AI/ML Concepts
- **Vector Embeddings**: How neural networks create semantic representations of images and text
- **HuggingFace CLIP Model**: Multi-modal AI using Vision Transformer architecture that understands both vision and language
- **Vision Transformer (ViT)**: Advanced architecture for processing images as sequences of patches
- **Cosine Similarity**: Mathematical foundation for semantic similarity search
- **Vector Databases**: Efficient storage and retrieval of high-dimensional embeddings

### Backend Development
- **FastAPI**: Modern async Python web framework
- **MongoDB Atlas Vector Search**: Cloud-native vector database operations
- **Async/Await Patterns**: Non-blocking I/O for better performance
- **Pydantic Validation**: Type-safe request/response handling
- **Production Best Practices**: Health checks, error handling, performance monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (3.11+ recommended)
- **uv** package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- MongoDB Atlas account with vector search enabled
- 8GB+ RAM for loading HuggingFace CLIP model
- (Optional) CUDA-capable GPU or Apple Silicon (MPS) for faster inference

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd backend
```

2. **Create virtual environment (with uv)**
```bash
uv venv  # Creates .venv automatically
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Alternative with standard Python:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies (using uv - recommended)**
```bash
uv sync  # Installs dependencies from pyproject.toml
```

**Alternative with pip:**
```bash
pip install -e .  # Installs from pyproject.toml
```

4. **Set up environment variables**
```bash
# Create .env file with your MongoDB connection
touch .env
# Add the following to .env:
# MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
```

5. **Run the server**
```bash
python main.py
# Or with uvicorn directly:
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

6. **Access the API**
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 📚 API Endpoints

### Health Check
```http
GET /health
```
Comprehensive health check that verifies all system components are operational.

**Response Example:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:30:00",
  "components": {
    "mongodb": {
      "status": "healthy",
      "latency_ms": 23.5,
      "total_products": 44424,
      "products_with_embeddings": 44424,
      "coverage_percentage": 100.0
    },
    "clip_model": {
      "status": "healthy",
      "latency_ms": 156.3,
      "device": "cuda",
      "embedding_dimension": 768,
      "model": "openai/clip-vit-large-patch14"
    },
    "vector_search": {
      "status": "healthy",
      "latency_ms": 45.2,
      "index_name": "vector_index",
      "test_results_found": true
    }
  },
  "message": "All systems operational"
}
```

### Text-to-Image Search
```http
POST /search/text
```
Find fashion items by natural language description using HuggingFace CLIP's text embeddings.

**Request Body:**
```json
{
  "search_query": "red floral summer dress",
  "number_of_results": 20,
  "similarity_threshold": 0.5,
  "filters": {
    "gender": "Women",
    "master_category": "Apparel"
  }
}
```

**Response:**
```json
{
  "success": true,
  "query_type": "text-to-image",
  "results_count": 20,
  "total_results": 20,
  "results": [
    {
      "id": "prod_12345",
      "name": "Floral Print Summer Dress",
      "brand": "Fashion Brand",
      "image": "https://example.com/dress.jpg",
      "similarity_score": 0.8923,
      "article_type": "Dress",
      "master_category": "Apparel",
      "sub_category": "Dress",
      "base_colour": "Red",
      "gender": "Women",
      "season": "Summer",
      "year": 2024,
      "usage": "Casual",
      "description": "Beautiful floral pattern dress perfect for summer"
    }
  ],
  "performance": {
    "embedding_generation": 0.156,
    "vector_search": 0.234,
    "result_formatting": 0.022,
    "total_time": 0.412
  },
  "similarity_stats": {
    "max": 0.8923,
    "min": 0.5234,
    "average": 0.6789,
    "std_dev": 0.1234
  },
  "message": "Found 20 products matching 'red floral summer dress'"
}
```

### Image-to-Image Search (JSON)
```http
POST /search/image
```
Find visually similar products using JSON request body. Supports base64-encoded images or image URLs.

**Option 1: Base64 Image**
```json
{
  "input_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "number_of_results": 20,
  "exclude_reference": true
}
```

**Option 2: Image URL**
```json
{
  "image_url": "https://example.com/reference-image.jpg",
  "number_of_results": 20,
  "exclude_reference": true
}
```

**Note:** For file uploads, use the `/search/image/v2` endpoint described below.

### Image-to-Image Search (File Upload)
```http
POST /search/image/v2
```
Find visually similar products by uploading an image file. This endpoint uses multipart form data.

**Request Parameters:**
- `file` (required): Image file to search with
- `number_of_results` (optional, default=10): Number of results to return
- `exclude_reference` (optional, default=true): Exclude the reference image from results

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/search/image/v2" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg" \
  -F "number_of_results=20" \
  -F "exclude_reference=true"
```

**Python Example:**
```python
import requests

with open('image.jpg', 'rb') as f:
    files = {'file': f}
    data = {
        'number_of_results': 20,
        'exclude_reference': True
    }
    response = requests.post(
        'http://localhost:8000/search/image/v2',
        files=files,
        data=data
    )
    print(response.json())
```

**Response (same format for both image endpoints):**
```json
{
  "success": true,
  "query_type": "image-to-image",
  "results_count": 20,
  "results": [
    {
      "id": "prod_67890",
      "name": "Similar Style Dress",
      "brand": "Another Brand",
      "image": "https://example.com/similar-dress.jpg",
      "similarity_score": 0.9234,
      "article_type": "Dress",
      "gender": "Women",
      "base_colour": "Red",
      "usage": "Casual"
    }
  ],
  "performance": {
    "embedding_generation": 0.189,
    "vector_search": 0.267,
    "result_formatting": 0.018,
    "total_time": 0.474
  },
  "similarity_stats": {
    "max": 0.9234,
    "min": 0.6123,
    "average": 0.7456,
    "std_dev": 0.0987
  },
  "image_source": "file_upload",
  "message": "Found 20 visually similar products"
}
```

### Statistics
```http
GET /stats
```
Get comprehensive database and system statistics.

**Response Example:**
```json
{
  "database_stats": {
    "total_products": 44424,
    "products_with_embeddings": 44424,
    "embedding_coverage": 100.0,
    "database_name": "fashion_semantic_search",
    "collection_name": "products"
  },
  "system_info": {
    "vector_index": "vector_index",
    "embedding_model": "openai/clip-vit-large-patch14",
    "embedding_dimensions": 768,
    "device": "cuda",
    "similarity_metric": "cosine"
  },
  "data_distribution": {
    "top_categories": [
      {"category": "Apparel", "count": 23456},
      {"category": "Footwear", "count": 8765},
      {"category": "Accessories", "count": 12203}
    ],
    "gender_distribution": [
      {"gender": "Women", "count": 18234},
      {"gender": "Men", "count": 15678},
      {"gender": "Unisex", "count": 10512}
    ]
  }
}
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client App    │────▶│   FastAPI Server │────▶│ MongoDB Atlas   │
│  (Frontend/API) │     │                  │     │ Vector Search   │
└─────────────────┘     │  ┌────────────┐ │     └─────────────────┘
                        │  │ CLIP Model │ │
                        │  └────────────┘ │
                        └──────────────────┘
```

### Key Components

1. **FastAPI Application**
   - Async request handling for better performance
   - Automatic API documentation generation
   - Request validation with Pydantic models
   - Support for both JSON and multipart form data

2. **HuggingFace CLIP Model Integration**
   - Model: `openai/clip-vit-large-patch14` (Vision Transformer Large)
   - Framework: HuggingFace Transformers library
   - Generates 768-dimensional embeddings
   - Supports both text and image inputs
   - Multi-device support: GPU acceleration (CUDA/MPS) and CPU fallback
   - Automatic device selection for optimal performance

3. **MongoDB Atlas Vector Search**
   - Stores fashion product metadata and embeddings
   - Performs approximate nearest neighbor search
   - Uses cosine similarity for matching
   - Scales to millions of products

### Key Dependencies

- **FastAPI**: Modern async web framework for building APIs
- **HuggingFace Transformers**: Deep learning library with pre-trained CLIP model
- **PyTorch**: Deep learning framework for model execution
- **Motor**: Async MongoDB driver for Python
- **Pillow**: Image processing library for handling uploads
- **Loguru**: Advanced logging for better debugging and monitoring

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:
```env
# MongoDB Configuration (Required)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/

```

### MongoDB Vector Index Configuration

Create a vector search index named `vector_index` in MongoDB Atlas:

```json
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "image_embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

## 🧪 Testing

### Run Tests
```bash
uv run ruff check
```

### Test with cURL

**Text Search:**
```bash
curl -X POST "http://localhost:8000/search/text" \
  -H "Content-Type: application/json" \
  -d '{
    "search_query": "blue denim jeans", 
    "number_of_results": 5,
    "similarity_threshold": 0.5
  }'
```

**Image Search with Base64 (JSON):**
```bash
# First encode your image to base64
base64_image=$(base64 -i test_image.jpg)

curl -X POST "http://localhost:8000/search/image" \
  -H "Content-Type: application/json" \
  -d "{
    \"input_image\": \"data:image/jpeg;base64,$base64_image\",
    \"number_of_results\": 5
  }"
```

**Image Search with File Upload:**
```bash
curl -X POST "http://localhost:8000/search/image/v2" \
  -F "file=@test_image.jpg" \
  -F "number_of_results=5" \
  -F "exclude_reference=true"
```

## 📈 Performance Optimization

### 1. **Request Optimization**
- Results are capped at 100 to prevent performance degradation
- Use `similarity_threshold` to filter low-quality matches early
- Apply metadata filters to reduce search space
- Image exclusion uses 0.99 similarity threshold to filter duplicates

### 2. **Model Optimization**
- CLIP model is loaded once at startup
- GPU acceleration automatically detected (CUDA/MPS)
- Model set to evaluation mode for faster inference
- Efficient batch processing for embeddings

### 3. **Database Optimization**

- Vector index with optimized `numCandidates` setting
- Connection pooling with Motor async driver
- Projection to return only necessary fields
- Aggregation pipeline optimization

### 4. **Response Optimization**

- Pagination for large result sets
- Performance metrics included in responses
- Statistical summaries for result quality assessment
- Efficient result formatting

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'torch'"**
   - Solution: Ensure dependencies are installed with `uv sync` or `pip install -e .`
   - For GPU support, you may need to install PyTorch separately: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

2. **"CUDA out of memory"**
   - Solution: Use CPU mode by setting `DEVICE=cpu` in `.env`
   - Or reduce batch size in code

3. **"Vector search returns no results"**
   - Check if products have `image_embedding` field
   - Verify vector index named `vector_index` exists
   - Ensure embedding dimensions match (768)
   - Check similarity threshold isn't too high

4. **Slow performance**
   - Use GPU acceleration if available
   - Reduce `number_of_results` in requests
   - Ensure MongoDB Atlas cluster is in same region
   - Check `numCandidates` in vector search query

5. **Image upload issues**
   - Ensure image is in supported format (JPEG, PNG)
   - Check file size limits
   - Verify multipart form data headers





## 📊 Monitoring

The API provides detailed performance metrics in every response:

- `embedding_generation`: Time to generate embeddings
- `vector_search`: Time for database search
- `result_formatting`: Time to format results
- `total_time`: End-to-end request time

Monitor these metrics to identify bottlenecks and optimize performance.

## 📚 Learning Resources

### Vector Databases
- [MongoDB Atlas Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-search/vector-search/)
- [Understanding Vector Embeddings](https://www.pinecone.io/learn/vector-embeddings/)
- [Approximate Nearest Neighbor Search](https://www.pinecone.io/learn/what-is-similarity-search/)

### HuggingFace CLIP Model
- [OpenAI CLIP Paper](https://arxiv.org/abs/2103.00020) - Original research paper
- [HuggingFace CLIP Model](https://huggingface.co/openai/clip-vit-large-patch14) - Model card and documentation
- [HuggingFace Transformers CLIP](https://huggingface.co/docs/transformers/model_doc/clip) - Implementation guide
- [Vision Transformer (ViT) Paper](https://arxiv.org/abs/2010.11929) - Understanding the vision architecture
- [Understanding Multimodal AI](https://huggingface.co/blog/vision_language_pretraining) - Technical blog

### FastAPI
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Async Python for Web Development](https://realpython.com/async-io-python/)
- [Building Production-Ready APIs](https://testdriven.io/blog/fastapi-best-practices/)


---

**Built with ❤️ for learning semantic search, HuggingFace CLIP, and modern AI applications**
