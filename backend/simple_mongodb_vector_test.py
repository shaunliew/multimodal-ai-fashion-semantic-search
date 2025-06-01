"""
Educational MongoDB Vector Search Tutorial

This simplified script teaches the fundamentals of MongoDB Atlas Vector Search
for semantic fashion image search. Focus on learning core concepts:

1. Text-to-Image Search: Natural language → Similar fashion items
2. Image-to-Image Search: Reference image → Visually similar items
3. MongoDB Aggregation Pipelines: Step-by-step vector search construction

Learning Goals:
- Understand $vectorSearch aggregation stage
- Learn similarity scoring with $meta
- Practice pipeline composition for complex queries
- Explore vector search parameters and their effects

Technical Stack:
- MongoDB Atlas Vector Search (cosine similarity)
- CLIP embeddings (768-dimensional vectors)
- Aggregation pipelines for semantic search
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional

import torch
from transformers import CLIPProcessor, CLIPModel
from pymongo import MongoClient
from loguru import logger
from dotenv import load_dotenv

class SimpleVectorSearchTutorial:
    """
    Educational vector search implementation focusing on core concepts.
    
    Learning: This class demonstrates the essential components of semantic
    search without the complexity of comprehensive testing frameworks.
    
    Core Concepts Covered:
    - Vector embedding generation with CLIP
    - MongoDB aggregation pipeline construction
    - Similarity scoring and result filtering
    - Cross-modal search (text ↔ image)
    """
    
    def __init__(self):
        """Initialize the tutorial environment."""
        load_dotenv()
        
        # MongoDB setup
        self.mongodb_uri = os.getenv("MONGODB_URI")
        if not self.mongodb_uri:
            raise ValueError("MONGODB_URI environment variable not set")
        
        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client["fashion_semantic_search"]
        self.collection = self.db["products"]
        
        # Use the known working index from previous discovery
        self.vector_index_name = "vector_index"
        
        # CLIP model setup for embedding generation
        self.device = self._get_device()
        logger.info(f"🔧 Using device: {self.device}")
        
        logger.info("📚 Loading CLIP model for embedding generation...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.model.eval()
        
        logger.success("✅ Tutorial environment initialized!")
    
    def _get_device(self) -> str:
        """Determine the best device for CLIP model inference."""
        if torch.backends.mps.is_available():
            return "mps"  # Apple Silicon GPU
        elif torch.cuda.is_available():
            return "cuda"  # NVIDIA GPU
        else:
            return "cpu"   # CPU fallback
    
    def generate_text_embedding(self, text: str) -> List[float]:
        """
        Generate CLIP embedding for text input.
        
        Learning: CLIP creates shared semantic space for text and images.
        This enables cross-modal search where text descriptions can find
        visually similar images, even if the text wasn't used to describe them.
        
        Process:
        1. Tokenize text input using CLIP's text processor
        2. Extract text features using CLIP's text encoder
        3. Normalize features for cosine similarity (unit vector)
        4. Convert to Python list for MongoDB storage
        
        Args:
            text: Natural language description (e.g., "red summer dress")
            
        Returns:
            768-dimensional normalized embedding vector
        """
        logger.info(f"🔤 Generating text embedding for: '{text}'")
        
        # Step 1: Process text through CLIP tokenizer
        # Learning: CLIP uses BPE tokenization similar to GPT models
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Step 2: Generate text features
        # Learning: CLIP's text encoder transforms tokens into semantic vectors
        with torch.no_grad():  # Disable gradients for inference
            text_features = self.model.get_text_features(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # Step 3: Normalize for cosine similarity
            # Learning: Normalized vectors enable cosine similarity = dot product
            # This makes similarity search more intuitive (0.0 = different, 1.0 = identical)
            normalized_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            # Step 4: Convert to Python list
            embedding = normalized_features.cpu().numpy().flatten().tolist()
        
        logger.info(f"📐 Generated {len(embedding)}D embedding (norm: {np.linalg.norm(embedding):.3f})")
        return embedding
    
    def text_to_image_search(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Demonstrate text-to-image semantic search using MongoDB aggregation.
        
        Learning: This is the core of semantic search - converting natural language
        into vector space and finding similar images. The aggregation pipeline
        performs approximate nearest neighbor search in high-dimensional space.
        
        MongoDB Aggregation Pipeline Breakdown:
        
        Stage 1 - $vectorSearch:
        - Converts vector similarity into database query
        - Uses approximate nearest neighbor algorithms (HNSW)
        - Returns documents sorted by similarity
        
        Stage 2 - $addFields:
        - Extracts similarity score from search metadata
        - Makes scores available for filtering and display
        
        Stage 3 - $project:
        - Selects only needed fields for response
        - Reduces network transfer and improves performance
        
        Args:
            query_text: Natural language description
            limit: Maximum results to return
            
        Returns:
            List of similar products with similarity scores
        """
        logger.info(f"\n🔍 TEXT-TO-IMAGE SEARCH TUTORIAL")
        logger.info(f"Query: '{query_text}'")
        logger.info(f"Target: Find fashion items matching this description")
        
        # Step 1: Convert text to embedding vector
        # Learning: This puts the text query into the same semantic space as image embeddings
        start_time = time.time()
        query_embedding = self.generate_text_embedding(query_text)
        embedding_time = time.time() - start_time
        
        # Step 2: Construct MongoDB aggregation pipeline
        # Learning: Aggregation pipelines process documents through sequential stages
        logger.info(f"\n📊 BUILDING AGGREGATION PIPELINE:")
        
        pipeline = [
            # Stage 1: Vector Similarity Search
            # Learning: $vectorSearch is MongoDB's vector similarity operator
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,     # Vector search index name
                    "path": "image_embedding",           # Field containing vectors
                    "queryVector": query_embedding,      # Our search vector
                    "numCandidates": limit * 10,         # Search space size (accuracy vs speed)
                    "limit": limit                       # Maximum results
                }
            },
            
            # Stage 2: Add Similarity Score
            # Learning: $meta extracts metadata from previous stages
            {
                "$addFields": {
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            },
            
            # Stage 3: Project Desired Fields
            # Learning: $project shapes the output, similar to SELECT in SQL
            {
                "$project": {
                    "name": 1,                           # Include product name
                    "brand": 1,                          # Include brand
                    "image": 1,                          # Include image URL
                    "year":1,                            # Include year
                    "usage":1,                            # Include usage
                    "gender":1,                           # Include gender
                    "season":1,                           # Include season
                    "article_type": 1,                   # Include article_type
                    "sub_category": 1,                   # Include sub_category
                    "master_category": 1,                # Include master_category
                    "description": 1,                    # Include description
                    "base_colour": 1,                     # Include base color
                    "similarity_score": 1,               # Include our calculated score
                    "_id": 0                             # Exclude MongoDB ObjectId
                }
            }
        ]
        
        # Log pipeline explanation
        logger.info(f"  Stage 1: $vectorSearch")
        logger.info(f"    - Index: {self.vector_index_name}")
        logger.info(f"    - Field: image_embedding")
        logger.info(f"    - Candidates: {limit * 10} (searches broader space for quality)")
        logger.info(f"    - Limit: {limit}")
        
        logger.info(f"  Stage 2: $addFields")
        logger.info(f"    - Extracts similarity score from search metadata")
        logger.info(f"    - Score range: 0.0 (different) to 1.0 (identical)")
        
        logger.info(f"  Stage 3: $project")
        logger.info(f"    - Selects fields: name, brand, image, year, usage, gender, season,")
        logger.info(f"      article_type, sub_category, master_category, description, base_colour, similarity_score")
        logger.info(f"    - Excludes: _id (MongoDB internal)")
        
        # Step 3: Execute the aggregation pipeline
        logger.info(f"\n⚡ EXECUTING PIPELINE:")
        search_start = time.time()
        
        try:
            results = list(self.collection.aggregate(pipeline))
            search_time = time.time() - search_start
            total_time = time.time() - start_time
            
            # Step 4: Analyze and display results
            logger.success(f"✅ Search completed successfully!")
            logger.info(f"📊 Performance:")
            logger.info(f"  - Embedding generation: {embedding_time:.3f}s")
            logger.info(f"  - Vector search: {search_time:.3f}s")
            logger.info(f"  - Total time: {total_time:.3f}s")
            
            if results:
                logger.info(f"\n🎯 FOUND {len(results)} SIMILAR PRODUCTS:")
                
                # Calculate similarity statistics
                scores = [r["similarity_score"] for r in results]
                avg_score = np.mean(scores)
                max_score = max(scores)
                min_score = min(scores)
                
                logger.info(f"📈 Similarity Range: {min_score:.3f} - {max_score:.3f} (avg: {avg_score:.3f})")
                
                # Display top results
                for i, product in enumerate(results[:3], 1):
                    logger.info(f"\n  {i}. {product['name']}")
                    logger.info(f"     Brand: {product.get('brand', 'N/A')}")
                    logger.info(f"     Article Type: {product.get('article_type', 'N/A')}")
                    logger.info(f"     Category: {product.get('master_category', 'N/A')} > {product.get('sub_category', 'N/A')}")
                    logger.info(f"     Gender: {product.get('gender', 'N/A')} | Season: {product.get('season', 'N/A')}")
                    logger.info(f"     Usage: {product.get('usage', 'N/A')} | Year: {product.get('year', 'N/A')}")
                    logger.info(f"     Color: {product.get('base_colour', 'N/A')}")
                    logger.info(f"     Description: {product.get('description', 'N/A')}")
                    logger.info(f"     Similarity: {product['similarity_score']:.3f}")
                    logger.info(f"     Image: {product.get('image', 'N/A')}")
                
                return results
            else:
                logger.warning("⚠️ No similar products found")
                logger.info("💡 Try adjusting the query or checking embedding coverage")
                return []
                
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            logger.info("🔧 Common issues:")
            logger.info("  - Vector index not found or misconfigured")
            logger.info("  - Embedding dimension mismatch")
            logger.info("  - MongoDB Atlas connection issues")
            return []
    
    def image_to_image_search(self, reference_product_name: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Demonstrate image-to-image similarity search using existing product embeddings.
        
        Learning: Image-to-image search finds products with similar visual characteristics:
        colors, patterns, styles, textures. This uses the same aggregation pipeline
        as text search, but with a pre-computed image embedding as the query vector.
        
        Key Difference from Text Search:
        - Query vector comes from existing product embedding (not generated)
        - Searches for visual similarity rather than semantic description match
        - Often produces higher similarity scores (images vs. cross-modal)
        
        Pipeline Stages:
        1. Get reference embedding from existing product
        2. Use same $vectorSearch aggregation as text search
        3. Filter out the reference product from results
        
        Args:
            reference_product_name: Name pattern to find reference product
            limit: Maximum similar products to return
            
        Returns:
            List of visually similar products
        """
        logger.info(f"\n🖼️ IMAGE-TO-IMAGE SEARCH TUTORIAL")
        logger.info(f"Goal: Find products visually similar to a reference image")
        
        # Step 1: Get reference product and its embedding
        logger.info(f"\n🎯 FINDING REFERENCE PRODUCT:")
        
        # Find a reference product (use pattern matching if name provided)
        if reference_product_name:
            reference_query = {"name": {"$regex": reference_product_name, "$options": "i"}}
            logger.info(f"Searching for products matching: '{reference_product_name}'")
        else:
            reference_query = {}
            logger.info(f"Selecting random product as reference")
        
        # Get reference product with embedding
        reference_product = self.collection.find_one(
            {**reference_query, "image_embedding": {"$exists": True}},
            {"name": 1, "brand": 1, "image": 1, "image_embedding": 1}
        )
        
        if not reference_product:
            logger.error("❌ No reference product found")
            return []
        
        reference_embedding = reference_product["image_embedding"]
        reference_id = reference_product["_id"]
        
        logger.success(f"✅ Reference product: {reference_product['name']}")
        logger.info(f"📐 Using {len(reference_embedding)}D embedding")
        logger.info(f"🖼️ Image: {reference_product.get('image', 'N/A')}")
        
        # Step 2: Build similarity search pipeline
        logger.info(f"\n📊 BUILDING IMAGE SIMILARITY PIPELINE:")
        
        pipeline = [
            # Stage 1: Vector Similarity Search
            # Learning: Same as text search, but using image embedding as query
            {
                "$vectorSearch": {
                    "index": self.vector_index_name,
                    "path": "image_embedding", 
                    "queryVector": reference_embedding,
                    "numCandidates": limit * 20,        # More candidates for image search
                    "limit": limit + 1                  # +1 to account for reference product
                }
            },
            
            # Stage 2: Add similarity score
            {
                "$addFields": {
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            },
            
            # Stage 3: Filter out reference product
            # Learning: $match filters documents, similar to WHERE in SQL
            {
                "$match": {
                    "_id": {"$ne": reference_id}        # Not equal to reference ID
                }
            },
            
            # Stage 4: Limit to desired count
            {
                "$limit": limit
            },
            
            # Stage 5: Project fields
            {
                "$project": {
                    "name": 1,                           # Include product name
                    "brand": 1,                          # Include brand
                    "image": 1,                          # Include image URL
                    "year":1,                            # Include year
                    "usage":1,                            # Include usage
                    "gender":1,                           # Include gender
                    "season":1,                           # Include season
                    "article_type": 1,                   # Include article_type
                    "sub_category": 1,                   # Include sub_category
                    "master_category": 1,                # Include master_category
                    "description": 1,                    # Include description
                    "base_colour": 1,                     # Include base color
                    "similarity_score": 1,               # Include our calculated score
                    "_id": 0                             # Exclude MongoDB ObjectId
                }
            }
        ]
        
        # Pipeline explanation
        logger.info(f"  Stage 1: $vectorSearch (same as text search)")
        logger.info(f"    - Using existing image embedding as query vector")
        logger.info(f"    - More candidates ({limit * 20}) for higher quality")
        
        logger.info(f"  Stage 2: $addFields (extract similarity score)")
        
        logger.info(f"  Stage 3: $match (filter results)")
        logger.info(f"    - Excludes reference product from results")
        logger.info(f"    - Uses $ne (not equal) operator")
        
        logger.info(f"  Stage 4: $limit (final result count)")
        logger.info(f"  Stage 5: $project (select fields)")
        logger.info(f"    - Includes: name, brand, image, year, usage, gender, season,")
        logger.info(f"      article_type, sub_category, master_category, description, base_colour, similarity_score")
        logger.info(f"    - Excludes: _id (MongoDB internal)")
        
        # Step 3: Execute pipeline
        logger.info(f"\n⚡ EXECUTING IMAGE SIMILARITY SEARCH:")
        start_time = time.time()
        
        try:
            results = list(self.collection.aggregate(pipeline))
            search_time = time.time() - start_time
            
            logger.success(f"✅ Image similarity search completed!")
            logger.info(f"⚡ Search time: {search_time:.3f}s")
            
            if results:
                logger.info(f"\n🎯 FOUND {len(results)} VISUALLY SIMILAR PRODUCTS:")
                
                # Image similarity typically produces higher scores
                scores = [r["similarity_score"] for r in results]
                avg_score = np.mean(scores)
                max_score = max(scores)
                
                logger.info(f"📈 Similarity Range: {min(scores):.3f} - {max_score:.3f} (avg: {avg_score:.3f})")
                logger.info(f"💡 Note: Image similarities are typically higher than text-to-image")
                
                # Display results with reference comparison
                for i, product in enumerate(results[:3], 1):
                    logger.info(f"\n  {i}. {product['name']}")
                    logger.info(f"     Brand: {product.get('brand', 'N/A')}")
                    logger.info(f"     Article Type: {product.get('article_type', 'N/A')}")
                    logger.info(f"     Category: {product.get('master_category', 'N/A')} > {product.get('sub_category', 'N/A')}")
                    logger.info(f"     Gender: {product.get('gender', 'N/A')} | Season: {product.get('season', 'N/A')}")
                    logger.info(f"     Usage: {product.get('usage', 'N/A')} | Year: {product.get('year', 'N/A')}")
                    logger.info(f"     Color: {product.get('base_colour', 'N/A')}")
                    logger.info(f"     Description: {product.get('description', 'N/A')}")
                    logger.info(f"     Similarity: {product['similarity_score']:.3f}")
                    logger.info(f"     Image: {product.get('image', 'N/A')}")
                
                return results
            else:
                logger.warning("⚠️ No similar images found")
                return []
                
        except Exception as e:
            logger.error(f"❌ Image similarity search failed: {e}")
            return []
    
    def run_tutorial(self):
        """
        Run the complete vector search tutorial with both search types.
        
        Learning: This demonstrates the two fundamental types of semantic search
        in e-commerce and content discovery applications.
        """
        logger.info("🎓 MONGODB VECTOR SEARCH TUTORIAL")
        logger.info("=" * 60)
        logger.info("Learning objectives:")
        logger.info("1. Understand MongoDB aggregation pipelines for vector search")
        logger.info("2. Compare text-to-image vs image-to-image search")
        logger.info("3. Learn pipeline optimization techniques")
        logger.info("4. Explore similarity scoring and result interpretation")
        
        try:
            # Tutorial 1: Text-to-Image Search
            logger.info(f"\n{'='*20} TUTORIAL 1: TEXT-TO-IMAGE SEARCH {'='*20}")
            
            # Try different types of fashion queries
            text_queries = [
                "blue denim jacket",
                "elegant black dress", 
                "comfortable running shoes"
            ]
            
            for query in text_queries:
                results = self.text_to_image_search(query, limit=3)
                time.sleep(1)  # Brief pause between searches
            
            # Tutorial 2: Image-to-Image Search
            logger.info(f"\n{'='*20} TUTORIAL 2: IMAGE-TO-IMAGE SEARCH {'='*20}")
            
            # Search for products similar to items containing certain words
            image_references = [
                "jacket",
                "dress", 
                "shoes"
            ]
            
            for ref in image_references:
                results = self.image_to_image_search(reference_product_name=ref, limit=3)
                time.sleep(1)  # Brief pause between searches
            
            # Summary
            logger.info(f"\n{'='*20} TUTORIAL SUMMARY {'='*20}")
            logger.success("🎉 Vector search tutorial completed!")
            logger.info("\n🧠 Key learnings:")
            logger.info("1. $vectorSearch: Core aggregation stage for similarity search")
            logger.info("2. $addFields + $meta: Extract similarity scores")  
            logger.info("3. $match: Filter results (exclude reference, apply thresholds)")
            logger.info("4. $project: Shape output for application needs")
            logger.info("5. numCandidates: Balance between accuracy and speed")
            
            logger.info("\n🚀 Next steps:")
            logger.info("- Experiment with different similarity thresholds")
            logger.info("- Try hybrid search (vector + metadata filters)")
            logger.info("- Optimize numCandidates for your use case")
            logger.info("- Implement result caching for better performance")
            
        except Exception as e:
            logger.error(f"❌ Tutorial failed: {e}")
        
        finally:
            self.close()
    
    def close(self):
        """Clean up resources."""
        if self.client:
            self.client.close()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("🧹 Tutorial resources cleaned up")

def main():
    """Run the educational vector search tutorial."""
    try:
        tutorial = SimpleVectorSearchTutorial()
        tutorial.run_tutorial()
    except Exception as e:
        logger.error(f"Tutorial initialization failed: {e}")
        logger.info("💡 Make sure MONGODB_URI is set in your .env file")

if __name__ == "__main__":
    main() 