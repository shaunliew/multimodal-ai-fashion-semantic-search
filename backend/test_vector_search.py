"""
Vector Search Testing Script for Fashion Image Embeddings

This script demonstrates:
- MongoDB Atlas Vector Search validation
- CLIP-based semantic similarity testing
- Cross-modal search (text query → image results)
- Image-to-image similarity search
- Vector search index performance analysis
- Embedding quality assessment

Learning: Validates that CLIP embeddings enable semantic fashion search
across different modalities (text descriptions, visual similarity).

Technical: Tests cosine similarity search using MongoDB's $vectorSearch
aggregation pipeline with CLIP ViT-Large 768-dimensional embeddings.
"""

import os
import time
import json
import requests
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pymongo import MongoClient
from loguru import logger
from dotenv import load_dotenv

class VectorSearchTester:
    """
    Test suite for validating fashion image embeddings and vector search.
    
    Learning: Comprehensive testing ensures vector search quality and helps
    identify issues with embedding generation, index configuration, or query performance.
    
    Technical: Tests both text-to-image and image-to-image semantic similarity
    using MongoDB Atlas Vector Search with cosine similarity metrics.
    """
    
    def __init__(self, 
                 mongodb_uri: str, 
                 database_name: str, 
                 collection_name: str,
                 model_name: str = "openai/clip-vit-large-patch14"):
        """
        Initialize vector search testing environment.
        
        Learning: Uses the same CLIP model as embedding generation to ensure
        query embeddings are in the same semantic space as stored embeddings.
        
        Args:
            mongodb_uri: MongoDB Atlas connection string
            database_name: Database containing fashion products
            collection_name: Collection with image embeddings
            model_name: CLIP model for generating query embeddings
        """
        self.mongodb_uri = mongodb_uri
        self.database_name = database_name
        self.collection_name = collection_name
        self.model_name = model_name
        
        # Will be set after index validation
        self.working_index_name = None
        
        # Setup device (same as embedding generation)
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Testing on device: {self.device}")
        
        # Load CLIP model for query embedding generation
        logger.info(f"Loading CLIP model for testing: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # MongoDB connection
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Configure test logging
        self.test_results = {
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "performance_metrics": {}
        }
        
        logger.success("Vector search tester initialized successfully")
    
    def set_working_index(self, index_name: str):
        """
        Directly set the working index name to bypass validation.
        
        Learning: Once you know the working index name, you can set it directly
        to skip the discovery process and make testing faster.
        
        Args:
            index_name: The known working vector search index name
        """
        self.working_index_name = index_name
        logger.info(f"📝 Working index set to: {index_name}")
    
    def validate_vector_index(self) -> Dict[str, Any]:
        """
        Validate that vector search index exists and is properly configured.
        
        Learning: MongoDB Atlas Vector Search indexes are managed through Atlas UI/API,
        not through standard pymongo methods. We'll test the index by attempting a search.
        
        Returns:
            Index validation results with configuration details
        """
        logger.info("🔍 Validating vector search index configuration...")
        
        try:
            # Test vector search functionality instead of checking index metadata
            # Learning: Since Atlas vector indexes aren't accessible via pymongo,
            # we validate by testing if vector search operations work
            
            # Get a sample embedding to test with
            sample_doc = self.collection.find_one(
                {"image_embedding": {"$exists": True}},
                {"image_embedding": 1}
            )
            
            if not sample_doc:
                return {
                    "index_exists": False,
                    "error": "No documents with embeddings found for testing",
                    "suggestion": "Ensure you have documents with image_embedding field"
                }
            
            test_embedding = sample_doc["image_embedding"]
            
            # First try the known working index name
            working_index = "vector_index"
            
            try:
                # Test the known working index first
                test_pipeline = [
                    {
                        "$vectorSearch": {
                            "index": working_index,
                            "path": "image_embedding",
                            "queryVector": test_embedding,
                            "numCandidates": 5,
                            "limit": 1
                        }
                    },
                    {"$limit": 1}
                ]
                
                # Try to execute the pipeline
                test_results = list(self.collection.aggregate(test_pipeline))
                
                if test_results:
                    # The known working index works!
                    logger.success(f"✅ Using known working vector index: {working_index}")
                    self.working_index_name = working_index
                    return {
                        "index_exists": True,
                        "index_name": working_index,
                        "validation_method": "direct_known_index",
                        "status": "Vector search is operational"
                    }
                else:
                    logger.warning(f"⚠️ Known index '{working_index}' returned no results, trying alternatives...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Known index '{working_index}' failed: {e}, trying alternatives...")
            
            # Fallback: Try different possible index names if the known one doesn't work
            logger.info("🔍 Testing alternative index names...")
            index_names_to_test = [
                "image_embedding",        # User thought this was the name
                "image_embedding_index",
                "default",
                "image_vector_search",
                "semantic_search_index"
            ]
            
            for index_name in index_names_to_test:
                try:
                    # Test this index name
                    pipeline = [
                        {
                            "$vectorSearch": {
                                "index": index_name,
                                "path": "image_embedding",
                                "queryVector": test_embedding,
                                "numCandidates": 5,
                                "limit": 1
                            }
                        },
                        {"$limit": 1}
                    ]
                    
                    # Try to execute the pipeline
                    test_results = list(self.collection.aggregate(pipeline))
                    
                    # If we get here without error, the index works
                    if test_results:
                        working_index = index_name
                        logger.success(f"✅ Found alternative working vector index: {index_name}")
                        break
                        
                except Exception as e:
                    # This index name doesn't work, try the next one
                    logger.debug(f"Index '{index_name}' failed: {e}")
                    continue
            
            if test_results:
                self.working_index_name = working_index
                return {
                    "index_exists": True,
                    "index_name": working_index,
                    "validation_method": "fallback_search",
                    "status": "Vector search is operational"
                }
            else:
                return {
                    "index_exists": False,
                    "error": "No working vector search index found",
                    "tested_names": ["vector_index"] + index_names_to_test,
                    "suggestion": """
                    Create a vector search index in MongoDB Atlas:
                    1. Go to Database Deployments → Browse Collections
                    2. Select your collection → Search Indexes tab
                    3. Create Search Index → JSON Editor
                    4. Use this configuration:
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
                    """
                }
            
        except Exception as e:
            logger.error(f"❌ Index validation failed: {e}")
            return {
                "index_exists": False,
                "error": str(e),
                "suggestion": "Check MongoDB Atlas connection and vector search index configuration"
            }
    
    def validate_embeddings_data(self) -> Dict[str, Any]:
        """
        Validate embedding data quality and completeness.
        
        Learning: Good embeddings should have consistent dimensions,
        reasonable value ranges, and proper normalization for cosine similarity.
        
        Returns:
            Embedding data validation results
        """
        logger.info("📊 Validating embedding data quality...")
        
        try:
            # Check total products and embedding coverage
            total_products = self.collection.count_documents({})
            with_embeddings = self.collection.count_documents({"image_embedding": {"$exists": True}})
            
            if with_embeddings == 0:
                logger.error("❌ No products have embeddings!")
                return {
                    "has_embeddings": False,
                    "error": "No embeddings found in collection"
                }
            
            # Analyze embedding quality
            sample_docs = list(self.collection.find(
                {"image_embedding": {"$exists": True}}, 
                {"image_embedding": 1, "name": 1}
            ).limit(10))
            
            embedding_dims = [len(doc["image_embedding"]) for doc in sample_docs]
            embedding_samples = [doc["image_embedding"] for doc in sample_docs]
            
            # Calculate embedding statistics
            # Learning: Well-normalized embeddings should have mean ~0, std ~0.3-0.5
            all_values = [val for embedding in embedding_samples for val in embedding]
            mean_value = np.mean(all_values)
            std_value = np.std(all_values)
            min_value = np.min(all_values)
            max_value = np.max(all_values)
            
            # Check L2 norms (should be ~1.0 for normalized embeddings)
            norms = [np.linalg.norm(embedding) for embedding in embedding_samples]
            mean_norm = np.mean(norms)
            
            validation_results = {
                "has_embeddings": True,
                "total_products": total_products,
                "products_with_embeddings": with_embeddings,
                "coverage_percentage": (with_embeddings / total_products) * 100,
                "embedding_dimension": embedding_dims[0] if embedding_dims else 0,
                "dimension_consistency": len(set(embedding_dims)) == 1,
                "statistics": {
                    "mean_value": round(mean_value, 4),
                    "std_value": round(std_value, 4),
                    "min_value": round(min_value, 4),
                    "max_value": round(max_value, 4),
                    "mean_norm": round(mean_norm, 4),
                    "expected_norm": "~1.0 (normalized embeddings)"
                },
                "sample_products": [doc["name"] for doc in sample_docs[:3]]
            }
            
            # Quality assessment
            if validation_results["dimension_consistency"] and abs(mean_norm - 1.0) < 0.1:
                logger.success("✅ Embedding quality looks good!")
            else:
                logger.warning("⚠️ Embedding quality issues detected")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"❌ Embedding validation failed: {e}")
            return {"has_embeddings": False, "error": str(e)}
    
    def generate_text_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """
        Generate CLIP embedding for text search query.
        
        Learning: Text queries are embedded into the same semantic space
        as images, enabling cross-modal fashion search.
        
        Args:
            query_text: Natural language description of desired fashion item
            
        Returns:
            768-dimensional query embedding for vector search
        """
        try:
            inputs = self.processor(text=[query_text], return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                text_features = self.model.get_text_features(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                
                # Normalize for cosine similarity
                normalized_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                embedding = normalized_features.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            return None
    
    def generate_image_query_embedding(self, image_url: str) -> Optional[List[float]]:
        """
        Generate CLIP embedding for image-based search query.
        
        Learning: Image-to-image search finds products with similar
        visual characteristics (style, color, pattern, etc.).
        
        Args:
            image_url: URL of reference image for similarity search
            
        Returns:
            768-dimensional image embedding for vector search
        """
        try:
            # Download and process image
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Process through CLIP
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                normalized_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                embedding = normalized_features.cpu().numpy().flatten().tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Image query embedding failed: {e}")
            return None
    
    def test_text_to_image_search(self, 
                                  query_text: str, 
                                  limit: int = 5,
                                  similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Test cross-modal search: text query → similar fashion images.
        
        Learning: This tests the core semantic search capability where
        natural language descriptions find visually similar products.
        
        Args:
            query_text: Fashion item description
            limit: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search results with performance metrics
        """
        logger.info(f"🔤 Testing text-to-image search: '{query_text}'")
        
        # Check if we have a working index
        if not self.working_index_name:
            return {
                "success": False, 
                "error": "No working vector search index available. Run index validation first."
            }
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.generate_text_query_embedding(query_text)
            if not query_embedding:
                return {"success": False, "error": "Failed to generate query embedding"}
            
            embedding_time = time.time() - start_time
            
            # Perform vector search
            search_start = time.time()
            
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.working_index_name,
                        "path": "image_embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 20,  # Search more candidates for quality
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
                        "name": 1,                           # Include product name
                        "brand": 1,                          # Include brand
                        "image": 1,                          # Include image URL
                        "year": 1,                           # Include year
                        "usage": 1,                          # Include usage
                        "gender": 1,                         # Include gender
                        "season": 1,                         # Include season
                        "article_type": 1,                   # Include article_type
                        "sub_category": 1,                   # Include sub_category
                        "master_category": 1,                # Include master_category
                        "description": 1,                    # Include description
                        "base_colour": 1,                    # Include base colour
                        "similarity_score": 1,               # Include our calculated score
                        "_id": 0                             # Exclude MongoDB ObjectId
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            # Filter results by similarity threshold after retrieval
            if similarity_threshold > 0.0:
                results = [r for r in results if r["similarity_score"] >= similarity_threshold]
            
            search_time = time.time() - search_start
            total_time = time.time() - start_time
            
            # Analyze results
            if results:
                avg_similarity = np.mean([r["similarity_score"] for r in results])
                max_similarity = max([r["similarity_score"] for r in results])
                min_similarity = min([r["similarity_score"] for r in results])
                
                logger.success(f"✅ Found {len(results)} similar products")
                logger.info(f"   Similarity range: {min_similarity:.3f} - {max_similarity:.3f}")
                
                return {
                    "success": True,
                    "query": query_text,
                    "results_count": len(results),
                    "results": results,
                    "performance": {
                        "embedding_time": round(embedding_time, 3),
                        "search_time": round(search_time, 3),
                        "total_time": round(total_time, 3)
                    },
                    "similarity_stats": {
                        "average": round(avg_similarity, 3),
                        "max": round(max_similarity, 3),
                        "min": round(min_similarity, 3)
                    }
                }
            else:
                logger.warning("⚠️ No results found")
                return {
                    "success": True,
                    "query": query_text,
                    "results_count": 0,
                    "results": [],
                    "performance": {
                        "total_time": round(total_time, 3)
                    },
                    "message": "No products found above similarity threshold"
                }
                
        except Exception as e:
            logger.error(f"❌ Text-to-image search failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_image_to_image_search(self, 
                                   reference_product_id: str = None,
                                   reference_image_url: str = None,
                                   limit: int = 5) -> Dict[str, Any]:
        """
        Test image-to-image similarity search.
        
        Learning: Visual similarity search finds products with similar
        colors, patterns, styles, and visual characteristics.
        
        Args:
            reference_product_id: MongoDB _id of reference product
            reference_image_url: External image URL for reference
            limit: Number of similar products to find
            
        Returns:
            Visual similarity search results
        """
        logger.info("🖼️ Testing image-to-image similarity search")
        
        # Check if we have a working index
        if not self.working_index_name:
            return {
                "success": False, 
                "error": "No working vector search index available. Run index validation first."
            }
        
        start_time = time.time()
        
        try:
            # Get reference embedding
            if reference_product_id:
                # Use existing product as reference
                ref_product = self.collection.find_one(
                    {"_id": reference_product_id},
                    {"image_embedding": 1, "name": 1, "image": 1}
                )
                
                if not ref_product or "image_embedding" not in ref_product:
                    return {"success": False, "error": "Reference product not found or missing embedding"}
                
                query_embedding = ref_product["image_embedding"]
                reference_info = {
                    "type": "product",
                    "name": ref_product['name'],
                    "image_url": ref_product.get('image', 'N/A')
                }
                
                logger.info(f"🔍 Reference: {ref_product['name']}")
                logger.info(f"🖼️ Reference Image: {ref_product.get('image', 'N/A')}")
                
            elif reference_image_url:
                # Generate embedding for external image
                query_embedding = self.generate_image_query_embedding(reference_image_url)
                if not query_embedding:
                    return {"success": False, "error": "Failed to generate embedding for reference image"}
                reference_info = {
                    "type": "external",
                    "image_url": reference_image_url
                }
                
                logger.info(f"🖼️ Reference Image: {reference_image_url}")
                
            else:
                return {"success": False, "error": "Must provide either product_id or image_url"}
            
            # Perform similarity search
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.working_index_name,
                        "path": "image_embedding", 
                        "queryVector": query_embedding,
                        "numCandidates": limit * 20,
                        "limit": limit + 1  # +1 to exclude reference if it's in results
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$project": {
                        "name": 1,                           # Include product name
                        "brand": 1,                          # Include brand
                        "image": 1,                          # Include image URL
                        "year": 1,                           # Include year
                        "usage": 1,                          # Include usage
                        "gender": 1,                         # Include gender
                        "season": 1,                         # Include season
                        "article_type": 1,                   # Include article_type
                        "sub_category": 1,                   # Include sub_category
                        "master_category": 1,                # Include master_category
                        "description": 1,                    # Include description
                        "base_colour": 1,                    # Include base colour
                        "similarity_score": 1,               # Include our calculated score
                        "_id": 1                             # Include _id for reference filtering
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            # Filter out reference product if present
            if reference_product_id:
                results = [r for r in results if str(r["_id"]) != str(reference_product_id)]
            
            results = results[:limit]  # Ensure we have the right number
            
            total_time = time.time() - start_time
            
            if results:
                avg_similarity = np.mean([r["similarity_score"] for r in results])
                logger.success(f"✅ Found {len(results)} visually similar products")
                
                # Log similar products with comprehensive fashion metadata
                logger.info("🔍 Similar Products Found:")
                for i, product in enumerate(results[:3], 1):
                    logger.info(f"\n  {i}. {product.get('name', 'Unknown')}")
                    logger.info(f"     Brand: {product.get('brand', 'N/A')}")
                    logger.info(f"     Article Type: {product.get('article_type', 'N/A')}")
                    logger.info(f"     Category: {product.get('master_category', 'N/A')} > {product.get('sub_category', 'N/A')}")
                    logger.info(f"     Gender: {product.get('gender', 'N/A')} | Season: {product.get('season', 'N/A')}")
                    logger.info(f"     Usage: {product.get('usage', 'N/A')} | Year: {product.get('year', 'N/A')}")
                    logger.info(f"     Color: {product.get('base_colour', 'N/A')}")
                    logger.info(f"     Description: {product.get('description', 'N/A')}")
                    logger.info(f"     Similarity: {product['similarity_score']:.3f}")
                    logger.info(f"     Image: {product.get('image', 'N/A')}")
                    logger.info(f"     �� View: {product.get('image', 'N/A')}")
                    print()
                
                return {
                    "success": True,
                    "reference": reference_info,
                    "results_count": len(results),
                    "results": results,
                    "performance": {
                        "total_time": round(total_time, 3)
                    },
                    "average_similarity": round(avg_similarity, 3)
                }
            else:
                logger.warning("⚠️ No similar images found")
                return {
                    "success": True,
                    "reference": reference_info,
                    "results_count": 0,
                    "results": [],
                    "message": "No similar products found"
                }
                
        except Exception as e:
            logger.error(f"❌ Image-to-image search failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite for vector search functionality.
        
        Learning: Systematic testing validates all aspects of semantic search:
        infrastructure, data quality, and search performance across different queries.
        
        Returns:
            Complete test results with pass/fail status and recommendations
        """
        logger.info("🧪 Starting comprehensive vector search test suite...")
        
        test_suite_results = {
            "test_timestamp": time.time(),
            "infrastructure_tests": {},
            "functionality_tests": {},
            "performance_tests": {},
            "overall_status": "unknown"
        }
        
        # 1. Infrastructure Tests
        logger.info("\n📋 Phase 1: Infrastructure Validation")
        
        index_validation = self.validate_vector_index()
        test_suite_results["infrastructure_tests"]["vector_index"] = index_validation
        
        data_validation = self.validate_embeddings_data()
        test_suite_results["infrastructure_tests"]["embedding_data"] = data_validation
        
        # Store working index name for subsequent tests
        if index_validation.get("index_exists") and "index_name" in index_validation:
            self.working_index_name = index_validation["index_name"]
            logger.info(f"📝 Using vector index: {self.working_index_name}")
        else:
            logger.warning("⚠️ No working vector index found - searches will fail")
        
        # 2. Functionality Tests
        logger.info("\n🔍 Phase 2: Search Functionality Tests")
        
        test_queries = [
            "blue denim jacket",
            "red summer dress",
            "black leather boots",
            "white cotton t-shirt",
            "elegant evening gown"
        ]
        
        text_search_results = []
        for query in test_queries:
            result = self.test_text_to_image_search(query, limit=3)
            text_search_results.append(result)
            time.sleep(0.5)  # Rate limiting
        
        test_suite_results["functionality_tests"]["text_to_image"] = text_search_results
        
        # Test image-to-image search with a random product
        random_product = self.collection.find_one(
            {"image_embedding": {"$exists": True}},
            {"_id": 1}
        )
        
        if random_product:
            image_search_result = self.test_image_to_image_search(
                reference_product_id=random_product["_id"],
                limit=3
            )
            test_suite_results["functionality_tests"]["image_to_image"] = image_search_result
        
        # 3. Performance Analysis
        logger.info("\n⚡ Phase 3: Performance Analysis")
        
        # Analyze response times
        successful_searches = [r for r in text_search_results if r.get("success")]
        if successful_searches:
            response_times = [r["performance"]["total_time"] for r in successful_searches]
            avg_response_time = np.mean(response_times)
            max_response_time = max(response_times)
            
            test_suite_results["performance_tests"] = {
                "average_response_time": round(avg_response_time, 3),
                "max_response_time": round(max_response_time, 3),
                "total_searches_tested": len(successful_searches),
                "performance_rating": "excellent" if avg_response_time < 0.5 else "good" if avg_response_time < 1.0 else "needs_improvement"
            }
        
        # 4. Overall Assessment
        logger.info("\n📊 Phase 4: Overall Assessment")
        
        # Calculate pass/fail status
        infrastructure_ok = (
            index_validation.get("index_exists", False) and
            data_validation.get("has_embeddings", False) and
            data_validation.get("coverage_percentage", 0) > 50
        )
        
        functionality_ok = (
            len([r for r in text_search_results if r.get("success") and r.get("results_count", 0) > 0]) >= 3
        )
        
        if infrastructure_ok and functionality_ok:
            test_suite_results["overall_status"] = "PASS"
            logger.success("🎉 Vector search is working correctly!")
        elif infrastructure_ok:
            test_suite_results["overall_status"] = "PARTIAL"
            logger.warning("⚠️ Infrastructure OK, but search quality needs improvement")
        else:
            test_suite_results["overall_status"] = "FAIL"
            logger.error("❌ Vector search setup needs attention")
        
        return test_suite_results
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """
        Generate human-readable test report.
        
        Learning: Clear reporting helps identify issues and validate
        that semantic search is working as expected.
        
        Args:
            test_results: Results from comprehensive test suite
            
        Returns:
            Formatted test report string
        """
        report = "\n" + "="*80 + "\n"
        report += "🧪 VECTOR SEARCH TEST REPORT\n"
        report += "="*80 + "\n\n"
        
        # Overall Status
        status = test_results["overall_status"]
        status_icon = "🎉" if status == "PASS" else "⚠️" if status == "PARTIAL" else "❌"
        report += f"Overall Status: {status_icon} {status}\n\n"
        
        # Infrastructure Tests
        report += "📋 INFRASTRUCTURE TESTS\n"
        report += "-" * 30 + "\n"
        
        index_test = test_results["infrastructure_tests"]["vector_index"]
        if index_test.get("index_exists"):
            report += f"✅ Vector Index: {index_test['index_name']}\n"
        else:
            report += f"❌ Vector Index: Missing or misconfigured\n"
            report += f"   Error: {index_test.get('error', 'Unknown')}\n"
        
        data_test = test_results["infrastructure_tests"]["embedding_data"]
        if data_test.get("has_embeddings"):
            coverage = data_test["coverage_percentage"]
            report += f"✅ Embeddings: {data_test['products_with_embeddings']:,} products ({coverage:.1f}% coverage)\n"
            report += f"   Dimensions: {data_test['embedding_dimension']}\n"
            report += f"   Quality: Mean norm = {data_test['statistics']['mean_norm']}\n"
        else:
            report += f"❌ Embeddings: No embeddings found\n"
        
        # Functionality Tests
        report += f"\n🔍 FUNCTIONALITY TESTS\n"
        report += "-" * 30 + "\n"
        
        text_tests = test_results["functionality_tests"]["text_to_image"]
        successful_text_searches = [t for t in text_tests if t.get("success") and t.get("results_count", 0) > 0]
        
        report += f"Text-to-Image Search: {len(successful_text_searches)}/{len(text_tests)} queries successful\n"
        
        for test in successful_text_searches[:3]:  # Show top 3
            similarity = test.get("similarity_stats", {}).get("max", 0)
            report += f"  ✅ '{test['query']}' → {test['results_count']} results (max similarity: {similarity:.3f})\n"
        
        if "image_to_image" in test_results["functionality_tests"]:
            img_test = test_results["functionality_tests"]["image_to_image"]
            if img_test.get("success") and img_test.get("results_count", 0) > 0:
                report += f"✅ Image-to-Image Search: {img_test['results_count']} similar products found\n"
            else:
                report += f"❌ Image-to-Image Search: No results\n"
        
        # Performance
        if "performance_tests" in test_results:
            perf = test_results["performance_tests"]
            report += f"\n⚡ PERFORMANCE\n"
            report += "-" * 30 + "\n"
            report += f"Average Response Time: {perf['average_response_time']}s\n"
            report += f"Performance Rating: {perf['performance_rating'].title()}\n"
        
        # Recommendations
        report += f"\n💡 RECOMMENDATIONS\n"
        report += "-" * 30 + "\n"
        
        if status == "PASS":
            report += "✅ Vector search is working excellently!\n"
            report += "✅ Ready for production semantic fashion search\n"
            report += "📈 Consider testing with more diverse queries\n"
        elif status == "PARTIAL":
            report += "⚠️ Basic functionality works but needs optimization\n"
            report += "🔧 Consider improving embedding coverage\n"
            report += "🎯 Fine-tune similarity thresholds\n"
        else:
            report += "🚨 Vector search needs immediate attention\n"
            if not index_test.get("index_exists"):
                report += "🔧 Create vector search index in MongoDB Atlas\n"
            if not data_test.get("has_embeddings"):
                report += "🔧 Generate embeddings for product images\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report
    
    def close(self):
        """Cleanup resources."""
        if self.client:
            self.client.close()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Vector search tester closed")

def main():
    """Main function to run vector search validation tests."""
    load_dotenv()
    
    # Configuration
    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        logger.error("MONGODB_URI environment variable not set")
        return
        
    DATABASE_NAME = "fashion_semantic_search"
    COLLECTION_NAME = "products"
    
    # Initialize tester
    tester = VectorSearchTester(
        mongodb_uri=MONGODB_URI,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME,
        model_name="openai/clip-vit-large-patch14"
    )
    
    # Optimization: Set known working index directly to skip discovery
    # Comment this out if you want to test index discovery
    tester.set_working_index("vector_index")
    
    try:
        logger.info("🚀 Starting Vector Search Validation Tests...")
        
        # Run comprehensive test suite
        test_results = tester.run_comprehensive_tests()
        
        # Generate and save report
        report = tester.generate_test_report(test_results)
        print(report)
        
        # Save detailed results
        timestamp = int(time.time())
        results_file = f"test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        logger.info(f"📊 Detailed test results saved to: {results_file}")
        
        # Quick demo searches if tests passed
        if test_results["overall_status"] in ["PASS", "PARTIAL"]:
            logger.info("\n🎯 Running Quick Demo Searches...")
            
            demo_queries = [
                "casual blue jeans",
                "elegant black dress",
                "comfortable running shoes"
            ]
            
            for query in demo_queries:
                result = tester.test_text_to_image_search(query, limit=2)
                if result.get("success") and result.get("results_count", 0) > 0:
                    print(f"\n🔍 Query: '{query}'")
                    for i, product in enumerate(result["results"][:2]):
                        print(f"\n  {i+1}. {product['name']}")
                        print(f"     Brand: {product.get('brand', 'N/A')}")
                        print(f"     Article Type: {product.get('article_type', 'N/A')}")
                        print(f"     Category: {product.get('master_category', 'N/A')} > {product.get('sub_category', 'N/A')}")
                        print(f"     Gender: {product.get('gender', 'N/A')} | Season: {product.get('season', 'N/A')}")
                        print(f"     Usage: {product.get('usage', 'N/A')} | Year: {product.get('year', 'N/A')}")
                        print(f"     Color: {product.get('base_colour', 'N/A')}")
                        print(f"     Description: {product.get('description', 'N/A')}")
                        print(f"     Similarity: {product['similarity_score']:.3f}")
                        print(f"     Image: {product['image']}")
                        print(f"     🔗 View: {product['image']}")
                        print()
        
        # Demo image-to-image search
        if test_results["overall_status"] in ["PASS", "PARTIAL"]:
            logger.info("\n🖼️ Running Image-to-Image Demo Search...")
            
            # Get a random product as reference
            random_ref_product = tester.collection.find_one(
                {"image_embedding": {"$exists": True}},
                {"_id": 1, "name": 1, "image": 1}
            )
            
            if random_ref_product:
                result = tester.test_image_to_image_search(
                    reference_product_id=random_ref_product["_id"], 
                    limit=3
                )
                if result.get("success") and result.get("results_count", 0) > 0:
                    print(f"\n🖼️ Image-to-Image Search Demo")
                    print(f"📸 Reference Product: {random_ref_product.get('name', 'Unknown')}")
                    print(f"🔗 Reference Image: {random_ref_product.get('image', 'N/A')}")
                    print(f"👀 View Reference: {random_ref_product.get('image', 'N/A')}")
                    print(f"\n🔍 Found {result['results_count']} similar products:")
                    
                    for i, product in enumerate(result["results"][:3]):
                        print(f"\n  {i+1}. {product['name']}")
                        print(f"     Brand: {product.get('brand', 'N/A')}")
                        print(f"     Article Type: {product.get('article_type', 'N/A')}")
                        print(f"     Category: {product.get('master_category', 'N/A')} > {product.get('sub_category', 'N/A')}")
                        print(f"     Gender: {product.get('gender', 'N/A')} | Season: {product.get('season', 'N/A')}")
                        print(f"     Usage: {product.get('usage', 'N/A')} | Year: {product.get('year', 'N/A')}")
                        print(f"     Color: {product.get('base_colour', 'N/A')}")
                        print(f"     Description: {product.get('description', 'N/A')}")
                        print(f"     Similarity: {product['similarity_score']:.3f}")
                        print(f"     Image: {product['image']}")
                        print(f"     🔗 View: {product['image']}")
                        print()
        
        if test_results["overall_status"] == "PASS":
            logger.success("\n🎉 Congratulations! Your vector search is working perfectly!")
            logger.info("🚀 Ready to build your semantic fashion search application!")
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        
    finally:
        tester.close()

if __name__ == "__main__":
    main() 