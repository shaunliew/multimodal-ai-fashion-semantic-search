"""
FastAPI Backend for Semantic Fashion Image Search

This module provides production-ready API endpoints for semantic image search
using MongoDB Atlas Vector Search and CLIP embeddings.

Learning Concepts:
- Vector similarity search with MongoDB Atlas
- Cross-modal search (text → image, image → image)
- FastAPI async patterns for AI workloads
- Efficient image handling and base64 encoding
- Response pagination and optimization
- Comprehensive error handling

Technical Stack:
- FastAPI: Modern async web framework
- MongoDB Atlas: Vector database for similarity search
- CLIP: Multi-modal AI model for embeddings
- Pydantic: Data validation and serialization
"""

import base64
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from transformers import CLIPModel, CLIPProcessor

# Load environment variables
load_dotenv()

# ===========================
# Pydantic Models
# ===========================


class HealthCheckResponse(BaseModel):
    """
    Health check response model.

    Learning: Health checks verify that all system components
    (database, models, indices) are functioning correctly.
    """

    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, dict[str, Any]] = Field(
        ..., description="Status of individual components"
    )
    message: str = Field(..., description="Human-readable status message")


class TextSearchRequest(BaseModel):
    """
    Request model for text-to-image semantic search.

    Learning: Text queries are converted to CLIP embeddings
    enabling cross-modal search in the same semantic space as images.
    """

    search_query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language description of desired fashion item",
        example="red summer dress with floral pattern",
    )

    number_of_results: int = Field(
        default=10, ge=1, le=100, description="Number of similar products to return", example=20
    )

    similarity_threshold: float | None = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0.0-1.0). Higher = more similar",
        example=0.5,
    )

    filters: dict[str, Any] | None = Field(
        default=None,
        description="Additional filters (e.g., gender, category, brand)",
        example={"gender": "Women", "master_category": "Apparel"},
    )

    @field_validator("search_query")
    @classmethod
    def validate_query(cls, v):
        """
        Learning: Query validation prevents issues that can cause
        poor vector search results or potential security risks.
        """
        if not v.strip():
            raise ValueError("Search query cannot be empty or whitespace only")

        # Security: Prevent potential injection attempts
        if len(v.split()) > 100:
            raise ValueError("Query too long. Keep descriptions concise for better results.")

        return v.strip()


class ImageSearchRequest(BaseModel):
    """
    Request model for image-to-image similarity search.

    Learning: Supports both base64-encoded images and URLs
    for flexibility in different client implementations.
    """

    input_image: str | None = Field(
        None,
        description="Base64-encoded image data",
        example="data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    )

    image_url: str | None = Field(
        None, description="URL of reference image", example="https://example.com/fashion/dress.jpg"
    )

    number_of_results: int = Field(
        default=10, ge=1, le=100, description="Number of similar products to return", example=20
    )

    exclude_reference: bool = Field(
        default=True, description="Whether to exclude the reference image from results"
    )

    @field_validator("input_image")
    @classmethod
    def validate_base64_image(cls, v, info: ValidationInfo):
        """Validate base64 image format."""
        if v and info.data.get("image_url"):
            raise ValueError("Provide either input_image or image_url, not both")

        if v:
            # Check for data URL format
            if v.startswith("data:image"):
                # Extract base64 part
                try:
                    header, data = v.split(",", 1)
                    return data
                except:
                    raise ValueError("Invalid data URL format")

        return v

    @field_validator("image_url")
    @classmethod
    def validate_image_url(cls, v, info: ValidationInfo):
        """Validate image URL format."""
        if v and not v.startswith(("http://", "https://")):
            raise ValueError("Image URL must start with http:// or https://")
        return v


class FashionProduct(BaseModel):
    """
    Fashion product model with comprehensive metadata.

    Learning: Rich metadata enables filtering and provides
    context for understanding search results.
    """

    id: str = Field(..., description="Product unique identifier")
    name: str = Field(..., description="Product name")
    brand: str | None = Field(None, description="Brand name")
    image: str = Field(..., description="Product image URL")

    # Fashion-specific attributes
    article_type: str | None = Field(None, description="Type of article (e.g., Shirt, Dress)")
    master_category: str | None = Field(None, description="Main category")
    sub_category: str | None = Field(None, description="Sub-category")
    gender: str | None = Field(None, description="Target gender")
    base_colour: str | None = Field(None, description="Primary color")
    season: str | None = Field(None, description="Seasonal collection")
    year: int | None = Field(None, description="Year of release")
    usage: str | None = Field(None, description="Usage context (e.g., Casual, Formal)")

    # Search metadata
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity score to query (0.0-1.0)"
    )

    # Optional description
    description: str | None = Field(None, description="Product description")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "prod_12345",
                "name": "Floral Summer Dress",
                "brand": "Fashion Brand",
                "image": "https://example.com/images/dress.jpg",
                "article_type": "Dress",
                "master_category": "Apparel",
                "sub_category": "Dress",
                "gender": "Women",
                "base_colour": "Red",
                "season": "Summer",
                "year": 2024,
                "usage": "Casual",
                "similarity_score": 0.89,
                "description": "Beautiful red summer dress with floral pattern",
            }
        }
    )


class SearchResponse(BaseModel):
    """
    Unified response model for search endpoints.

    Learning: Consistent response structure across endpoints
    simplifies client implementation and error handling.
    """

    success: bool = Field(..., description="Whether the search was successful")
    query_type: str = Field(..., description="Type of search performed")
    results_count: int = Field(..., description="Number of results returned")
    total_results: int | None = Field(None, description="Total available results")
    results: list[FashionProduct] = Field(..., description="Search results")

    # Performance metrics
    performance: dict[str, float] = Field(..., description="Performance metrics in seconds")

    # Search quality metrics
    similarity_stats: dict[str, float] | None = Field(
        None, description="Statistical summary of similarity scores"
    )

    # Pagination info
    pagination: dict[str, Any] | None = Field(
        None, description="Pagination information for large result sets"
    )

    # Image search specific
    image_source: str | None = Field(None, description="Source of the image for image searches")

    message: str | None = Field(None, description="Additional information")


# ===========================
# Global Resources
# ===========================


class AppResources:
    """
    Application resources singleton.

    Learning: Centralized resource management ensures efficient
    use of connections and models across requests.
    """

    def __init__(self):
        self.mongodb_client: AsyncIOMotorClient | None = None
        self.database = None
        self.collection = None
        self.clip_model = None
        self.clip_processor = None
        self.device = None
        self.working_index_name = "vector_index"  # Known working index

    async def initialize(self):
        """Initialize all application resources."""
        try:
            # MongoDB setup
            mongodb_uri = os.getenv("MONGODB_URI")
            if not mongodb_uri:
                raise ValueError("MONGODB_URI environment variable not set")

            self.mongodb_client = AsyncIOMotorClient(mongodb_uri)
            self.database = self.mongodb_client["fashion_semantic_search"]
            self.collection = self.database["products"]

            # Test database connection
            await self.database.command("ping")
            logger.success("✅ MongoDB connection established")

            # CLIP model setup
            self.setup_clip_model()

        except Exception as e:
            logger.error(f"❌ Resource initialization failed: {e}")
            raise

    def setup_clip_model(self):
        """
        Initialize CLIP model for embedding generation.

        Learning: Using the same model architecture as the stored embeddings
        ensures query and document embeddings are in the same semantic space.
        """
        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        logger.info(f"🖥️ Using device: {self.device}")

        # Load CLIP model
        model_name = "openai/clip-vit-large-patch14"
        logger.info(f"📦 Loading CLIP model: {model_name}")

        self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        self.clip_model.eval()  # Set to evaluation mode

        logger.success("✅ CLIP model loaded successfully")

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        if self.mongodb_client:
            self.mongodb_client.close()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("🧹 Resources cleaned up")


# Initialize resources
resources = AppResources()

# ===========================
# FastAPI Application
# ===========================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Learning: Lifespan events ensure proper initialization
    and cleanup of resources like database connections and models.
    """
    # Startup
    logger.info("🚀 Starting Semantic Fashion Search API...")
    await resources.initialize()

    yield

    # Shutdown
    logger.info("🛑 Shutting down...")
    await resources.cleanup()


app = FastAPI(
    title="Semantic Fashion Image Search API",
    description="""
    Production-ready API for semantic fashion search using MongoDB Atlas Vector Search.
    
    Features:
    - Text-to-image search: Find fashion items by description
    - Image-to-image search: Find visually similar products
    - CLIP-based multi-modal embeddings
    - Optimized for large-scale fashion catalogs
    
    Learning: This API demonstrates modern AI-powered search combining
    vector databases with multi-modal deep learning models.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Helper Functions
# ===========================


async def generate_text_embedding(text: str) -> list[float]:
    """
    Generate CLIP embedding for text query.

    Learning: Text embeddings enable semantic search where
    "red dress" finds images containing red dresses even without
    exact keyword matches in metadata.

    Args:
        text: Natural language query

    Returns:
        768-dimensional embedding vector
    """
    try:
        inputs = resources.clip_processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(resources.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = resources.clip_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )

            # L2 normalization for cosine similarity
            normalized_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            embedding = normalized_features.cpu().numpy().flatten().tolist()

        return embedding

    except Exception as e:
        logger.error(f"Text embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate text embedding: {e!s}")


async def generate_image_embedding(image: Image.Image) -> list[float]:
    """
    Generate CLIP embedding for image.

    Learning: Image embeddings capture visual features like
    color, pattern, style, and composition for similarity search.

    Args:
        image: PIL Image object

    Returns:
        768-dimensional embedding vector
    """
    try:
        inputs = resources.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(resources.device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = resources.clip_model.get_image_features(**inputs)
            normalized_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            embedding = normalized_features.cpu().numpy().flatten().tolist()

        return embedding

    except Exception as e:
        logger.error(f"Image embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate image embedding: {e!s}")


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.

    Learning: Base64 encoding allows images to be transmitted
    as text in JSON, though it increases payload size by ~33%.

    Args:
        base64_string: Base64-encoded image data

    Returns:
        PIL Image object
    """
    try:
        # Remove data URL prefix if present
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        return image

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e!s}")


async def perform_vector_search(
    query_embedding: list[float],
    limit: int,
    similarity_threshold: float = 0.0,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Perform vector similarity search in MongoDB Atlas.

    Learning: MongoDB's $vectorSearch uses approximate nearest neighbor
    algorithms for efficient similarity search in high-dimensional space.

    Args:
        query_embedding: Query vector
        limit: Maximum results to return
        similarity_threshold: Minimum similarity score
        filters: Additional metadata filters

    Returns:
        List of similar products with metadata
    """
    # Build aggregation pipeline
    # Learning: Aggregation pipelines combine vector search
    # with filtering, projection, and sorting operations
    pipeline = [
        {
            "$vectorSearch": {
                "index": resources.working_index_name,
                "path": "image_embedding",
                "queryVector": query_embedding,
                "numCandidates": min(limit * 20, 10000),  # Search wider for quality
                "limit": limit,
            }
        },
        {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}},
    ]

    # Apply similarity threshold
    if similarity_threshold > 0.0:
        pipeline.append({"$match": {"similarity_score": {"$gte": similarity_threshold}}})

    # Apply additional filters
    if filters:
        pipeline.append({"$match": filters})

    # Project fields for response
    pipeline.append(
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "brand": 1,
                "image": 1,
                "year": 1,
                "usage": 1,
                "gender": 1,
                "season": 1,
                "article_type": 1,
                "sub_category": 1,
                "master_category": 1,
                "description": 1,
                "base_colour": 1,
                "similarity_score": 1,
            }
        }
    )

    # Execute search
    try:
        cursor = resources.collection.aggregate(pipeline)
        results = await cursor.to_list(length=limit)
        return results
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search operation failed: {e!s}")


def format_search_results(
    raw_results: list[dict[str, Any]], max_results: int = None
) -> list[FashionProduct]:
    """
    Format MongoDB results into API response format.

    Learning: Consistent formatting ensures predictable
    API responses regardless of database schema changes.

    Args:
        raw_results: Raw MongoDB documents
        max_results: Limit results if specified

    Returns:
        List of formatted FashionProduct objects
    """
    formatted_results = []

    # Apply result limit if specified
    results_to_process = raw_results[:max_results] if max_results else raw_results

    for doc in results_to_process:
        formatted_results.append(
            FashionProduct(
                id=str(doc["_id"]),
                name=doc.get("name", "Unknown Product"),
                brand=doc.get("brand"),
                image=doc.get("image", ""),
                article_type=doc.get("article_type"),
                master_category=doc.get("master_category"),
                sub_category=doc.get("sub_category"),
                gender=doc.get("gender"),
                base_colour=doc.get("base_colour"),
                season=doc.get("season"),
                year=doc.get("year"),
                usage=doc.get("usage"),
                similarity_score=round(doc.get("similarity_score", 0.0), 4),
                description=doc.get("description"),
            )
        )

    return formatted_results


# ===========================
# API Endpoints
# ===========================


@app.get("/", response_model=dict[str, str])
async def root():
    """
    Root endpoint with API information.

    Learning: A welcoming root endpoint helps developers
    quickly understand what the API does.
    """
    return {
        "message": "Welcome to Semantic Fashion Image Search API",
        "documentation": "/docs",
        "health_check": "/health",
        "version": "1.0.0",
    }


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check endpoint.

    Learning: Health checks should verify all critical components
    to ensure the service is fully operational before accepting traffic.

    Returns:
        Detailed health status of all system components
    """
    start_time = time.time()
    components = {}

    # Check MongoDB connection
    try:
        # Ping database
        await resources.database.command("ping")

        # Check collection access
        doc_count = await resources.collection.count_documents({})
        embedding_count = await resources.collection.count_documents(
            {"image_embedding": {"$exists": True}}
        )

        components["mongodb"] = {
            "status": "healthy",
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "total_products": doc_count,
            "products_with_embeddings": embedding_count,
            "coverage_percentage": round(
                (embedding_count / doc_count * 100) if doc_count > 0 else 0, 2
            ),
        }
    except Exception as e:
        components["mongodb"] = {"status": "unhealthy", "error": str(e)}

    # Check CLIP model
    try:
        model_check_start = time.time()

        # Test embedding generation
        test_embedding = await generate_text_embedding("test")

        components["clip_model"] = {
            "status": "healthy",
            "latency_ms": round((time.time() - model_check_start) * 1000, 2),
            "device": resources.device,
            "embedding_dimension": len(test_embedding),
            "model": "openai/clip-vit-large-patch14",
        }
    except Exception as e:
        components["clip_model"] = {"status": "unhealthy", "error": str(e)}

    # Check vector search index
    try:
        # Test vector search with sample embedding
        if components.get("clip_model", {}).get("status") == "healthy":
            search_start = time.time()
            test_results = await perform_vector_search(query_embedding=test_embedding, limit=1)

            components["vector_search"] = {
                "status": "healthy",
                "latency_ms": round((time.time() - search_start) * 1000, 2),
                "index_name": resources.working_index_name,
                "test_results_found": len(test_results) > 0,
            }
        else:
            components["vector_search"] = {
                "status": "unknown",
                "reason": "CLIP model unhealthy, cannot test vector search",
            }
    except Exception as e:
        components["vector_search"] = {"status": "unhealthy", "error": str(e)}

    # Overall health status
    all_healthy = all(comp.get("status") == "healthy" for comp in components.values())

    return HealthCheckResponse(
        status="healthy" if all_healthy else "unhealthy",
        timestamp=datetime.utcnow(),
        components=components,
        message="All systems operational" if all_healthy else "Some components are unhealthy",
    )


@app.post("/search/text", response_model=SearchResponse)
async def search_by_text(request: TextSearchRequest):
    """
    Text-to-image semantic search endpoint.

    Learning: This endpoint demonstrates cross-modal search where
    natural language queries find visually similar fashion items
    using CLIP's shared embedding space.

    Args:
        request: Search parameters including query text and filters

    Returns:
        Ranked list of similar fashion products

    Example:
        POST /search/text
        {
            "search_query": "red floral summer dress",
            "number_of_results": 20,
            "similarity_threshold": 0.5
        }
    """
    start_time = time.time()
    performance_metrics = {}

    try:
        # Generate text embedding
        logger.info(f"🔍 Text search: '{request.search_query}'")
        embedding_start = time.time()

        query_embedding = await generate_text_embedding(request.search_query)
        performance_metrics["embedding_generation"] = round(time.time() - embedding_start, 3)

        # Perform vector search
        search_start = time.time()

        # Optimization: Limit candidates for very large result requests
        # Learning: Searching too many candidates can impact performance
        # without significantly improving result quality
        effective_limit = min(request.number_of_results, 100)

        raw_results = await perform_vector_search(
            query_embedding=query_embedding,
            limit=effective_limit,
            similarity_threshold=request.similarity_threshold,
            filters=request.filters,
        )

        performance_metrics["vector_search"] = round(time.time() - search_start, 3)

        # Format results
        format_start = time.time()
        formatted_results = format_search_results(raw_results, request.number_of_results)
        performance_metrics["result_formatting"] = round(time.time() - format_start, 3)

        # Calculate statistics
        similarity_stats = None
        if formatted_results:
            scores = [r.similarity_score for r in formatted_results]
            similarity_stats = {
                "max": round(max(scores), 4),
                "min": round(min(scores), 4),
                "average": round(np.mean(scores), 4),
                "std_dev": round(np.std(scores), 4),
            }

        # Pagination info for large requests
        pagination = None
        if request.number_of_results > effective_limit:
            pagination = {
                "requested": request.number_of_results,
                "returned": len(formatted_results),
                "limit_applied": True,
                "reason": "Performance optimization for large result sets",
            }

        performance_metrics["total_time"] = round(time.time() - start_time, 3)

        return SearchResponse(
            success=True,
            query_type="text-to-image",
            results_count=len(formatted_results),
            total_results=len(raw_results) if raw_results else 0,
            results=formatted_results,
            performance=performance_metrics,
            similarity_stats=similarity_stats,
            pagination=pagination,
            message=f"Found {len(formatted_results)} products matching '{request.search_query}'",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search operation failed: {e!s}")


@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(request: ImageSearchRequest):
    """
    Image-to-image similarity search endpoint (JSON only).

    Learning: This endpoint finds visually similar products using
    CLIP's image embeddings, enabling "shop the look" functionality.

    Supports two input methods via JSON:
    1. Base64-encoded image
    2. Image URL reference

    For file uploads, use the /search/image/v2 endpoint.

    Args:
        request: Search parameters including base64 image or URL

    Returns:
        Ranked list of visually similar products

    Example with base64:
        POST /search/image
        {
            "input_image": "data:image/jpeg;base64,...",
            "number_of_results": 20,
            "exclude_reference": true
        }

    Example with URL:
        POST /search/image
        {
            "image_url": "https://example.com/fashion/dress.jpg",
            "number_of_results": 20,
            "exclude_reference": true
        }
    """
    start_time = time.time()
    performance_metrics = {}

    try:
        # Determine image source
        image = None
        image_source = "unknown"

        # Priority 1: Base64 image
        if request.input_image:
            logger.info("📸 Processing base64 image")
            image = decode_base64_image(request.input_image)
            image_source = "base64"

        # Priority 2: Image URL
        elif request.image_url:
            logger.info(f"📸 Fetching image from URL: {request.image_url}")
            import requests

            response = requests.get(request.image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image_source = "url"

        else:
            raise HTTPException(
                status_code=400,
                detail="No image provided. Use 'input_image' for base64 or 'image_url' for URL. For file uploads, use /search/image/v2.",
            )

        # Generate image embedding
        embedding_start = time.time()
        query_embedding = await generate_image_embedding(image)
        performance_metrics["embedding_generation"] = round(time.time() - embedding_start, 3)

        # Perform vector search
        search_start = time.time()
        effective_limit = min(request.number_of_results, 100)

        # Get extra results to handle exclusion if needed
        search_limit = effective_limit + 1 if request.exclude_reference else effective_limit

        raw_results = await perform_vector_search(
            query_embedding=query_embedding, limit=search_limit
        )

        performance_metrics["vector_search"] = round(time.time() - search_start, 3)

        # Format results
        format_start = time.time()

        # Exclude perfect matches (likely the same image)
        if request.exclude_reference and raw_results:
            # Filter out results with similarity > 0.99 (likely same image)
            filtered_results = [r for r in raw_results if r.get("similarity_score", 0) < 0.99]
            formatted_results = format_search_results(filtered_results, request.number_of_results)
        else:
            formatted_results = format_search_results(raw_results, request.number_of_results)

        performance_metrics["result_formatting"] = round(time.time() - format_start, 3)

        # Calculate statistics
        similarity_stats = None
        if formatted_results:
            scores = [r.similarity_score for r in formatted_results]
            similarity_stats = {
                "max": round(max(scores), 4),
                "min": round(min(scores), 4),
                "average": round(np.mean(scores), 4),
                "std_dev": round(np.std(scores), 4),
            }

        # Pagination info
        pagination = None
        if request.number_of_results > effective_limit:
            pagination = {
                "requested": request.number_of_results,
                "returned": len(formatted_results),
                "limit_applied": True,
                "reason": "Performance optimization for large result sets",
            }

        performance_metrics["total_time"] = round(time.time() - start_time, 3)

        return SearchResponse(
            success=True,
            query_type="image-to-image",
            results_count=len(formatted_results),
            total_results=len(raw_results) if raw_results else 0,
            results=formatted_results,
            performance=performance_metrics,
            similarity_stats=similarity_stats,
            pagination=pagination,
            image_source=image_source,
            message=f"Found {len(formatted_results)} visually similar products",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search operation failed: {e!s}")


@app.post("/search/image/v2", response_model=SearchResponse)
async def search_by_image_v2(
    file: UploadFile = File(...),
    number_of_results: int = Form(10),
    exclude_reference: bool = Form(True),
):
    """
    Alternative image-to-image search endpoint using Form fields.
    
    Learning: This approach is cleaner for multipart form data as FastAPI
    automatically parses form fields without needing JSON parsing.
    
    Args:
        file: Uploaded image file (required)
        number_of_results: Number of similar products to return
        exclude_reference: Whether to exclude the reference image from results
        
    Returns:
        Ranked list of visually similar products
        
    Example:
        curl -X 'POST' \
          'http://127.0.0.1:8000/search/image/v2' \
          -H 'accept: application/json' \
          -H 'Content-Type: multipart/form-data' \
          -F 'file=@image.jpg;type=image/jpeg' \
          -F 'number_of_results=20' \
          -F 'exclude_reference=true'
    """
    start_time = time.time()
    performance_metrics = {}

    try:
        # Create request object from form fields
        parsed_request = ImageSearchRequest(
            number_of_results=number_of_results, exclude_reference=exclude_reference
        )

        # Process uploaded image
        logger.info(f"📸 Processing uploaded image: {file.filename}")
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_source = "file_upload"

        # Generate image embedding
        embedding_start = time.time()
        query_embedding = await generate_image_embedding(image)
        performance_metrics["embedding_generation"] = round(time.time() - embedding_start, 3)

        # Perform vector search
        search_start = time.time()
        effective_limit = min(parsed_request.number_of_results, 100)

        # Get extra results to handle exclusion if needed
        search_limit = effective_limit + 1 if parsed_request.exclude_reference else effective_limit

        raw_results = await perform_vector_search(
            query_embedding=query_embedding, limit=search_limit
        )

        performance_metrics["vector_search"] = round(time.time() - search_start, 3)

        # Format results
        format_start = time.time()

        # Exclude perfect matches (likely the same image)
        if parsed_request.exclude_reference and raw_results:
            # Filter out results with similarity > 0.99 (likely same image)
            filtered_results = [r for r in raw_results if r.get("similarity_score", 0) < 0.99]
            formatted_results = format_search_results(
                filtered_results, parsed_request.number_of_results
            )
        else:
            formatted_results = format_search_results(raw_results, parsed_request.number_of_results)

        performance_metrics["result_formatting"] = round(time.time() - format_start, 3)

        # Calculate statistics
        similarity_stats = None
        if formatted_results:
            scores = [r.similarity_score for r in formatted_results]
            similarity_stats = {
                "max": round(max(scores), 4),
                "min": round(min(scores), 4),
                "average": round(np.mean(scores), 4),
                "std_dev": round(np.std(scores), 4),
            }

        # Pagination info
        pagination = None
        if parsed_request.number_of_results > effective_limit:
            pagination = {
                "requested": parsed_request.number_of_results,
                "returned": len(formatted_results),
                "limit_applied": True,
                "reason": "Performance optimization for large result sets",
            }

        performance_metrics["total_time"] = round(time.time() - start_time, 3)

        return SearchResponse(
            success=True,
            query_type="image-to-image",
            results_count=len(formatted_results),
            total_results=len(raw_results) if raw_results else 0,
            results=formatted_results,
            performance=performance_metrics,
            similarity_stats=similarity_stats,
            pagination=pagination,
            image_source=image_source,
            message=f"Found {len(formatted_results)} visually similar products",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search operation failed: {e!s}")


@app.get("/stats", response_model=dict[str, Any])
async def get_statistics():
    """
    Get database and search statistics.

    Learning: Statistics help monitor system performance
    and identify potential issues with data quality.

    Returns:
        Database statistics and system information
    """
    try:
        # Get collection statistics
        total_products = await resources.collection.count_documents({})
        products_with_embeddings = await resources.collection.count_documents(
            {"image_embedding": {"$exists": True}}
        )

        # Get sample products for categories
        sample_categories = await resources.collection.aggregate(
            [
                {"$group": {"_id": "$master_category", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 10},
            ]
        ).to_list(length=10)

        # Get gender distribution
        gender_dist = await resources.collection.aggregate(
            [{"$group": {"_id": "$gender", "count": {"$sum": 1}}}, {"$sort": {"count": -1}}]
        ).to_list(length=10)

        return {
            "database_stats": {
                "total_products": total_products,
                "products_with_embeddings": products_with_embeddings,
                "embedding_coverage": round(
                    (products_with_embeddings / total_products * 100) if total_products > 0 else 0,
                    2,
                ),
                "database_name": "fashion_semantic_search",
                "collection_name": "products",
            },
            "system_info": {
                "vector_index": resources.working_index_name,
                "embedding_model": "openai/clip-vit-large-patch14",
                "embedding_dimensions": 768,
                "device": resources.device,
                "similarity_metric": "cosine",
            },
            "data_distribution": {
                "top_categories": [
                    {"category": cat["_id"], "count": cat["count"]}
                    for cat in sample_categories
                    if cat["_id"]
                ],
                "gender_distribution": [
                    {"gender": g["_id"], "count": g["count"]} for g in gender_dist if g["_id"]
                ],
            },
        }

    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {e!s}")


# ===========================
# Error Handlers
# ===========================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    Custom HTTP exception handler.

    Learning: Consistent error responses make debugging easier
    and improve the developer experience.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": "HTTPException",
                "status_code": exc.status_code,
                "detail": exc.detail,
            },
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    General exception handler for unexpected errors.

    Learning: Catching unexpected errors prevents exposing
    sensitive information while providing useful debug info.
    """
    logger.error(f"Unexpected error: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "type": "InternalServerError",
                "status_code": 500,
                "detail": "An unexpected error occurred. Please try again later.",
            },
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


if __name__ == "__main__":
    """
    Development server startup.
    
    Learning: For production, use proper ASGI server like
    uvicorn with multiple workers: uvicorn main:app --workers 4
    """
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
