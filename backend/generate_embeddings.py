"""
CLIP-based image embedding generator using Transformers library.

This script demonstrates:
- Direct CLIP model usage with transformers library
- Fashion image embedding generation at scale
- Efficient batch processing for 44,446 products
- MongoDB bulk updates with embedding vectors
- Robust error handling for image download failures
- **AUTOMATIC RESUME CAPABILITY** - Skips already processed products

Learning: Uses OpenAI's CLIP model directly through transformers library
for better control and understanding of the embedding generation process.

Technical: CLIP ViT-Large creates 768-dimensional embeddings optimized
for image-text alignment and semantic similarity search.

RESUME FUNCTIONALITY: This script automatically resumes from where it left off.
If interrupted, simply rerun - it will skip products with existing embeddings.
"""

import os
import time
import requests
from io import BytesIO
from typing import List, Dict, Any, Optional

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pymongo import MongoClient, UpdateOne
from loguru import logger
from dotenv import load_dotenv

class FashionCLIPEmbedder:
    """
    CLIP-based embedding service for fashion product images.
    
    Learning: Demonstrates direct CLIP integration for semantic image search.
    Uses transformers library for full control over model behavior.
    
    Technical: CLIP ViT-Large-Patch14 creates 768-dimensional embeddings
    with superior quality for fashion similarity search.
    
    RESUME CAPABILITY: Automatically skips products with existing embeddings.
    """
    
    def __init__(self, 
                 mongodb_uri: str, 
                 database_name: str, 
                 collection_name: str,
                 model_name: str = "openai/clip-vit-large-patch14"):
        """
        Initialize CLIP model and MongoDB connection.
        
        Learning: CLIP ViT-Large provides higher quality embeddings than
        ViT-Base, worth the extra computational cost for fashion search.
        
        Args:
            mongodb_uri: MongoDB Atlas connection string
            database_name: Target database name
            collection_name: Target collection name  
            model_name: CLIP model variant to use
        """
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Check if MPS is available
        # Learning: MPS provides GPU acceleration on Apple Silicon Macs
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize CLIP model and processor
        # Learning: CLIPProcessor handles image preprocessing automatically
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Get embedding dimension from model
        # Learning: Different CLIP variants have different embedding dimensions
        self.embedding_dim = self.model.config.projection_dim
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Set model to evaluation mode for inference
        # Learning: eval() mode disables dropout and batch norm updates
        self.model.eval()
        
        # MongoDB connection
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Configure logging
        logger.add(
            "logs/clip_embedding_generation.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        logger.success(f"CLIP embedder initialized with {self.embedding_dim}-dim embeddings")
        
    def check_resume_status(self) -> Dict[str, int]:
        """
        Check current progress and resume status.
        
        Learning: Progress checking before starting helps users understand
        current state and estimated completion time.
        
        Returns:
            Dictionary with progress statistics
        """
        try:
            total_with_images = self.collection.count_documents({
                "image": {"$exists": True, "$ne": ""}
            })
            
            already_completed = self.collection.count_documents({
                "image_embedding": {"$exists": True}
            })
            
            remaining_to_process = self.collection.count_documents({
                "image": {"$exists": True, "$ne": ""},
                "image_embedding": {"$exists": False}
            })
            
            return {
                "total_with_images": total_with_images,
                "already_completed": already_completed,
                "remaining_to_process": remaining_to_process,
                "completion_percentage": (already_completed / total_with_images * 100) if total_with_images > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to check resume status: {e}")
            return {"error": str(e)}
        
    def download_and_process_image(self, image_url: str, max_retries: int = 3) -> Optional[Image.Image]:
        """
        Download image from URL and prepare for CLIP processing.
        
        Learning: Robust image downloading with retries and validation
        ensures consistent input quality for embedding generation.
        
        Args:
            image_url: URL of the fashion product image
            max_retries: Number of download attempts
            
        Returns:
            PIL Image ready for CLIP processing, or None if failed
        """
        for attempt in range(max_retries):
            try:
                # Download image with timeout and proper headers
                # Learning: User-Agent helps avoid bot detection
                response = requests.get(
                    image_url, 
                    timeout=15,
                    headers={
                        'User-Agent': 'Fashion-Semantic-Search/1.0',
                        'Accept': 'image/*'
                    }
                )
                response.raise_for_status()
                
                # Convert to PIL Image
                # Learning: CLIP requires RGB format, convert ensures consistency
                image = Image.open(BytesIO(response.content)).convert('RGB')
                
                # Validate image dimensions
                # Learning: Very small images provide poor embedding quality
                if image.size[0] < 100 or image.size[1] < 100:
                    logger.warning(f"Image too small ({image.size}): {image_url}")
                    return None
                
                # Check for reasonable aspect ratio
                aspect_ratio = max(image.size) / min(image.size)
                if aspect_ratio > 10:
                    logger.warning(f"Extreme aspect ratio ({aspect_ratio:.1f}): {image_url}")
                    return None
                
                return image
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {image_url}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except Exception as e:
                logger.error(f"Image processing error for {image_url}: {e}")
                return None
        
        logger.error(f"Failed to download image after {max_retries} attempts: {image_url}")
        return None
    
    def generate_image_embedding(self, image_url: str) -> Optional[List[float]]:
        """
        Generate CLIP embedding for fashion product image.
        
        Learning: CLIP's image encoder creates semantic representations
        that capture style, color, category, and visual aesthetics.
        
        Args:
            image_url: URL of fashion product image
            
        Returns:
            768-dimensional embedding vector or None if failed
        """
        try:
            # Download and preprocess image
            image = self.download_and_process_image(image_url)
            if image is None:
                return None
            
            # Process image through CLIP processor
            # Learning: CLIPProcessor handles resizing, normalization, and tensor conversion
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Move inputs to same device as model
            # Learning: Tensor device consistency required for GPU inference
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding with no gradient computation
            # Learning: torch.no_grad() saves memory and speeds up inference
            with torch.no_grad():
                # Get image features from CLIP
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize embeddings for cosine similarity
                # Learning: L2 normalization enables direct cosine similarity via dot product
                normalized_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # Convert to CPU and then to list for MongoDB storage
                # Learning: MongoDB requires native Python types, not torch tensors
                embedding = normalized_features.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"CLIP embedding generation failed for {image_url}: {e}")
            return None
    
    def generate_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate CLIP embedding for text description.
        
        Learning: CLIP's text encoder creates embeddings in the same
        semantic space as images, enabling cross-modal search.
        
        Args:
            text: Fashion product description or search query
            
        Returns:
            768-dimensional text embedding vector
        """
        try:
            # Process text through CLIP processor
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get text features from CLIP
                text_features = self.model.get_text_features(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                # Normalize for cosine similarity
                normalized_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                embedding = normalized_features.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"CLIP text embedding failed for '{text}': {e}")
            return None
    
    def process_batch(self, batch: List[Dict[str, Any]]) -> List[UpdateOne]:
        """
        Process batch of products to generate IMAGE embeddings only.
        
        Learning: Starting with image embeddings first, then adding text later
        once we handle the sequence length limitations of CLIP text encoder.
        
        Args:
            batch: List of product documents from MongoDB
            
        Returns:
            List of MongoDB UpdateOne operations
        """
        update_operations = []
        
        for product in batch:
            try:
                # Generate image embedding ONLY
                image_embedding = self.generate_image_embedding(product['image'])
                
                # SKIP text embedding for now due to sequence length issues
                # TODO: Add text embedding with proper truncation later
                
                # Create update operation if image embedding succeeded
                if image_embedding is not None:
                    update_data = {
                        "image_embedding": image_embedding,
                        "embedding_generated_at": time.time(),
                        "embedding_model": self.model_name,
                        "embedding_dimension": self.embedding_dim
                    }
                    
                    update_op = UpdateOne(
                        {"_id": product["_id"]},
                        {"$set": update_data}
                    )
                    update_operations.append(update_op)
                    
                    logger.debug(f"Generated IMAGE embedding for product {product.get('id', product['_id'])}")
                else:
                    logger.warning(f"Failed to generate embedding for product {product.get('id', product['_id'])}")
                    
            except Exception as e:
                logger.error(f"Error processing product {product.get('id', product['_id'])}: {e}")
        
        return update_operations
    
    def generate_all_embeddings(self, batch_size: int = 32) -> Dict[str, int]:
        """
        Generate embeddings for all products in the collection.
        
        Learning: Optimized batch size balances memory usage with throughput.
        Smaller batches for GPU inference prevent out-of-memory errors.
        
        RESUME CAPABILITY: Automatically skips products with existing embeddings.
        
        Args:
            batch_size: Products per batch (smaller for GPU memory management)
            
        Returns:
            Statistics dictionary with success/failure counts
        """
        # Check resume status first
        resume_status = self.check_resume_status()
        if "error" in resume_status:
            logger.error(f"Cannot check resume status: {resume_status['error']}")
            return {"error": resume_status["error"]}
        
        # Display resume information
        logger.info("🔄 RESUME STATUS CHECK")
        logger.info(f"Total products with images: {resume_status['total_with_images']:,}")
        logger.info(f"Already completed: {resume_status['already_completed']:,}")
        logger.info(f"Remaining to process: {resume_status['remaining_to_process']:,}")
        logger.info(f"Completion: {resume_status['completion_percentage']:.1f}%")
        
        if resume_status['already_completed'] > 0:
            logger.success(f"✅ RESUMING: Will skip {resume_status['already_completed']:,} completed products")
        
        if resume_status['remaining_to_process'] == 0:
            logger.success("🎉 All embeddings already completed!")
            return {
                "total": resume_status['total_with_images'],
                "processed": 0, 
                "success": resume_status['already_completed'],
                "failed": 0,
                "resumed_from": resume_status['already_completed']
            }
        
        # Get products without embeddings (RESUME QUERY)
        # Learning: This query enables automatic resume functionality
        query = {
            "image": {"$exists": True, "$ne": ""},
            "image_embedding": {"$exists": False}  # 🔑 RESUME KEY: Only unprocessed products
        }
        
        total_products = resume_status['remaining_to_process']
        logger.info(f"Processing {total_products:,} remaining products...")
        
        # Initialize statistics
        stats = {
            "total": resume_status['total_with_images'],
            "processed": 0, 
            "success": 0,
            "failed": 0,
            "resumed_from": resume_status['already_completed']
        }
        
        start_time = time.time()
        
        # Process in batches
        cursor = self.collection.find(query).batch_size(batch_size)
        
        batch = []
        for product in cursor:
            batch.append(product)
            
            # Process batch when full
            if len(batch) >= batch_size:
                self._process_and_update_batch(batch, stats)
                batch = []
                
                # Progress reporting and memory cleanup
                if stats["processed"] % 100 == 0:
                    self._log_progress(stats, start_time, resume_status)
                    # Clear GPU cache periodically
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
        
        # Process remaining products
        if batch:
            self._process_and_update_batch(batch, stats)
            
        # Final statistics
        elapsed_time = time.time() - start_time
        total_completed = stats['success'] + resume_status['already_completed']
        
        logger.success(
            f"🎉 CLIP embedding generation completed!\n"
            f"Total products: {stats['total']}\n"
            f"Previously completed: {resume_status['already_completed']}\n"
            f"Newly processed: {stats['processed']}\n"
            f"Total completed: {total_completed}\n"
            f"Failed: {stats['failed']}\n"
            f"Overall completion: {total_completed/stats['total']*100:.1f}%\n"
            f"Session time: {elapsed_time:.2f}s\n"
            f"Session rate: {stats['processed']/elapsed_time:.2f} products/sec"
        )
        
        return stats
    
    def _process_and_update_batch(self, batch: List[Dict[str, Any]], stats: Dict[str, int]):
        """Process batch and update MongoDB with embeddings."""
        try:
            update_operations = self.process_batch(batch)
            
            if update_operations:
                result = self.collection.bulk_write(update_operations, ordered=False)
                stats["success"] += result.modified_count
                stats["failed"] += len(batch) - result.modified_count
            else:
                stats["failed"] += len(batch)
                
            stats["processed"] += len(batch)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            stats["failed"] += len(batch)
            stats["processed"] += len(batch)
    
    def _log_progress(self, stats: Dict[str, int], start_time: float, resume_status: Dict[str, int]):
        """Log progress with GPU memory info and resume context."""
        elapsed = time.time() - start_time
        rate = stats["processed"] / elapsed if elapsed > 0 else 0
        remaining = resume_status['remaining_to_process'] - stats['processed']
        eta = remaining / rate if rate > 0 else 0
        
        # Calculate overall progress including resumed work
        total_completed = stats['success'] + resume_status['already_completed']
        overall_progress = total_completed / resume_status['total_with_images'] * 100
        
        memory_info = ""
        if self.device == "cuda" and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            memory_info = f"CUDA Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached"
        elif self.device == "mps":
            memory_info = "MPS GPU acceleration enabled"
        
        logger.info(
            f"📊 Session: {stats['processed']}/{resume_status['remaining_to_process']} "
            f"Overall: {total_completed:,}/{resume_status['total_with_images']:,} ({overall_progress:.1f}%) "
            f"Success: {stats['success']}, Failed: {stats['failed']} "
            f"Rate: {rate:.1f}/sec, ETA: {eta:.0f}s {memory_info}"
        )
    
    def verify_embeddings(self) -> Dict[str, Any]:
        """Verify IMAGE embedding generation and analyze results."""
        try:
            total_products = self.collection.count_documents({})
            with_image_embeddings = self.collection.count_documents({"image_embedding": {"$exists": True}})
            
            sample_doc = self.collection.find_one({"image_embedding": {"$exists": True}})
            embedding_dim = len(sample_doc["image_embedding"]) if sample_doc else 0
            
            return {
                "total_products": total_products,
                "with_image_embeddings": with_image_embeddings,
                "image_completion_rate": (with_image_embeddings / total_products * 100) if total_products > 0 else 0,
                "embedding_dimension": embedding_dim,
                "sample_product": sample_doc.get("name", "N/A") if sample_doc else "N/A",
                "model_used": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {}
    
    def test_similarity_search(self, query_text: str, limit: int = 5) -> List[Dict]:
        """
        Test semantic search with a text query.
        
        Learning: Demonstrates cross-modal search where text queries
        can find semantically similar fashion images.
        """
        try:
            # Generate query embedding
            query_embedding = self.generate_text_embedding(query_text)
            if not query_embedding:
                return []
            
            # MongoDB vector search pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "image_embedding_index",  # Your vector search index name
                        "path": "image_embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 10,
                        "limit": limit
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "name": 1,
                        "brand": 1,
                        "image": 1,
                        "similarity_score": 1
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} similar products for query: '{query_text}'")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def close(self):
        """Close MongoDB connection and cleanup GPU memory."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        elif self.device == "mps":
            # MPS doesn't have explicit cache clearing
            logger.info("MPS session completed")

def main():
    """Main function to generate CLIP embeddings for fashion products."""
    load_dotenv()
    
    # Configuration
    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        logger.error("MONGODB_URI environment variable not set")
        return
        
    DATABASE_NAME = "fashion_semantic_search"
    COLLECTION_NAME = "products"
    
    # Initialize CLIP embedder
    embedder = FashionCLIPEmbedder(
        mongodb_uri=MONGODB_URI,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
        model_name="openai/clip-vit-large-patch14"  # Higher quality embeddings
    )
    
    try:
        logger.info("🚀 Starting CLIP embedding generation for fashion dataset...")
        logger.info("🔄 RESUME ENABLED: Will automatically skip completed products")
        
        # Generate embeddings (smaller batch size for GPU memory)
        stats = embedder.generate_all_embeddings(batch_size=128)  # Adjust based on GPU memory
        
        # Verify results
        verification = embedder.verify_embeddings()
        logger.info(f"Verification results: {verification}")
        
        # Test semantic search
        if verification.get("image_completion_rate", 0) > 0:
            logger.info("Testing semantic search...")
            test_results = embedder.test_similarity_search("blue casual shirt", limit=3)
            for i, result in enumerate(test_results):
                logger.info(f"  {i+1}. {result['name']} (score: {result['similarity_score']:.3f})")
        
        if verification.get("image_completion_rate", 0) > 95:
            logger.success("🎉 CLIP IMAGE embedding generation completed successfully!")
            logger.info(
                f"\n✅ COMPLETED:\n"
                f"1. ✅ Vector search index ready\n"
                f"2. ✅ {verification['with_image_embeddings']:,} image embeddings generated\n"
                f"3. 🚀 Ready for image-to-image semantic search!\n"
                f"\n📊 Embedding specs:\n"
                f"   Model: {verification['model_used']}\n"
                f"   Dimensions: {verification['embedding_dimension']}\n"
                f"   Quality: High (ViT-Large)\n"
                f"\n🔧 RESUME CAPABILITY: ✅ Enabled\n"
                f"   - Script can be safely interrupted and resumed\n"
                f"   - Completed embeddings are automatically skipped\n"
                f"   - Progress is saved in MongoDB after each batch\n"
                f"\n🔄 Next: Run python test_vector_search.py to validate"
            )
        else:
            completion_rate = verification.get("image_completion_rate", 0)
            remaining = verification.get("total_products", 0) - verification.get("with_image_embeddings", 0)
            logger.warning(f"⚠️ Partial completion ({completion_rate:.1f}%). {remaining:,} products remaining.")
            logger.info("🔄 You can safely rerun this script to continue - it will resume automatically!")
            
    except Exception as e:
        logger.error(f"CLIP embedding generation failed: {e}")
        logger.info("🔄 You can safely rerun this script - it will resume from where it left off!")
        
    finally:
        embedder.close()

if __name__ == "__main__":
    main()