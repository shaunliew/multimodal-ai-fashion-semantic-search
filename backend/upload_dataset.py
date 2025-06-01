"""
Fashion dataset uploader for MongoDB Atlas Vector Search.

This script demonstrates:
- Bulk operations for efficient database insertion
- JSON data processing and transformation
- MongoDB connection management with proper error handling
- Progress tracking for large dataset operations
- Schema design for vector search optimization

Learning: MongoDB Atlas provides flexible schema design where fields like
embeddings can be added later without restructuring existing documents.

Technical: Uses bulk operations to minimize network round trips and
improve insertion performance for large datasets.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from pymongo.collection import Collection
from loguru import logger
import time
from datetime import datetime, timezone

class FashionDatasetUploader:
    """
    Service for uploading fashion dataset to MongoDB Atlas.
    
    Learning: Demonstrates batch processing patterns for large datasets
    and MongoDB best practices for vector search document design.
    
    Technical: Uses bulk operations to achieve high throughput while
    maintaining data consistency and proper error handling.
    """
    
    def __init__(self, connection_string: str, database_name: str, collection_name: str):
        """
        Initialize MongoDB connection for dataset upload.
        
        Learning: MongoDB connection strings contain authentication,
        host information, and connection options for Atlas clusters.
        
        Args:
            connection_string: MongoDB Atlas connection string
            database_name: Target database name
            collection_name: Target collection name
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None
        
        # Configure logging for upload operations
        logger.add(
            "logs/dataset_upload.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
    
    def connect(self) -> bool:
        """
        Establish connection to MongoDB Atlas.
        
        Learning: Proper connection management prevents resource leaks
        and ensures reliable database operations.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Create MongoDB client with Atlas connection string
            # Learning: serverSelectionTimeoutMS prevents hanging on connection issues
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000  # 5 second timeout
            )
            
            # Test connection by running a simple command
            # Learning: Lazy connection means we need to trigger an operation
            self.client.admin.command('ping')
            
            # Get database and collection references
            database = self.client[self.database_name]
            self.collection = database[self.collection_name]
            
            logger.success(f"Connected to MongoDB Atlas database: {self.database_name}")
            return True
            
        except errors.ServerSelectionTimeoutError:
            logger.error("Failed to connect to MongoDB Atlas - check connection string and network")
            return False
        except errors.ConfigurationError as e:
            logger.error(f"MongoDB configuration error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return False
    
    def create_indexes(self) -> bool:
        """
        Create database indexes for optimized queries.
        
        Learning: Indexes improve query performance but should be created
        before large data insertion for better efficiency.
        
        Note: Vector search index will be created separately in Atlas UI
        after embedding generation.
        
        Returns:
            True if indexes created successfully
        """
        try:
            # Create compound index for common filter queries
            # Learning: Compound indexes support multiple query patterns
            self.collection.create_index([
                ("brand", 1),
                ("gender", 1),
                ("article_type", 1)
            ], name="brand_gender_type_idx")
            
            # Create index for color-based searches
            self.collection.create_index("base_colour", name="color_idx")
            
            # Create index for category browsing
            self.collection.create_index([
                ("master_category", 1),
                ("sub_category", 1)
            ], name="category_idx")
            
            logger.success("Database indexes created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False
    
    def extract_product_data(self, json_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract and transform product data from JSON structure.
        
        Learning: Data transformation normalizes inconsistent source data
        and prepares it for vector search operations.
        
        Args:
            json_data: Raw JSON data from style file
            
        Returns:
            Transformed document ready for MongoDB insertion
        """
        try:
            # Navigate to actual product data
            # Learning: Myntra API(e-commerce platform) wraps data in meta/notification structure
            if "data" not in json_data:
                logger.warning("JSON missing 'data' field - skipping")
                return None
            
            item = json_data["data"]
            
            # Extract fields with safe navigation and defaults
            # Learning: Safe navigation prevents KeyError exceptions
            # and ensures data consistency across varying source formats
            document = {
                "id": item.get("id"),  # Keep original ID for reference
                "name": item.get("productDisplayName", ""),
                "brand": item.get("brandName", ""),
                "usage": item.get("usage", ""),
                "year": item.get("year", 0),
                "season": item.get("season", ""),
                "base_colour": item.get("baseColour", ""),
                "article_type": item.get("articleType", {}).get("typeName", ""),
                "sub_category": item.get("subCategory", {}).get("typeName", ""),
                "master_category": item.get("masterCategory", {}).get("typeName", ""),
                "gender": item.get("gender", ""),
                "image": item.get("styleImages", {}).get("default", {}).get("imageURL", ""),
                
                # Additional useful fields for vector search context
                "price": item.get("price", 0),
                "description": self._extract_description(item),
                "colors": self._extract_colors(item),
                
                # Metadata for tracking and debugging
                "uploaded_at": datetime.now(timezone.utc),
                "source_file": f"{item.get('id', 'unknown')}.json"
            }
            
            # Validate required fields
            # Learning: Data validation prevents incomplete documents
            # that could cause issues during vector search operations
            if not document["name"] or not document["image"]:
                logger.warning(f"Product {document['id']} missing required fields - skipping")
                return None
            
            return document
            
        except Exception as e:
            logger.error(f"Error extracting product data: {e}")
            return None
    
    def _extract_description(self, item: Dict[str, Any]) -> str:
        """
        Extract and combine product descriptions for better embeddings.
        
        Learning: Rich text descriptions improve embedding quality
        by providing more semantic context for vector search.
        """
        descriptions = []
        
        # Get main description
        prod_desc = item.get("productDescriptors", {})
        if "description" in prod_desc:
            desc_html = prod_desc["description"].get("value", "")
            # Simple HTML tag removal for cleaner text
            desc_clean = desc_html.replace("<p>", "").replace("</p>", "").replace("<br>", " ")
            descriptions.append(desc_clean)
        
        # Add style notes for more context
        if "style_note" in prod_desc:
            style_html = prod_desc["style_note"].get("value", "")
            style_clean = style_html.replace("<p>", "").replace("</p>", "").replace("<br>", " ")
            descriptions.append(style_clean)
        
        return " ".join(descriptions).strip()
    
    def _extract_colors(self, item: Dict[str, Any]) -> List[str]:
        """
        Extract all color information for comprehensive search.
        
        Learning: Multiple color fields provide richer context
        for color-based semantic search queries.
        """
        colors = []
        
        base_color = item.get("baseColour", "")
        if base_color and base_color != "NA":
            colors.append(base_color)
        
        color1 = item.get("colour1", "")
        if color1 and color1 != "NA":
            colors.append(color1)
        
        color2 = item.get("colour2", "")
        if color2 and color2 != "NA":
            colors.append(color2)
        
        return list(set(colors))  # Remove duplicates
    
    def upload_dataset(self, dataset_path: str, batch_size: int = 1000) -> bool:
        """
        Upload fashion dataset from JSON files to MongoDB.
        
        Learning: Batch processing balances memory usage with performance.
        Large batches reduce network overhead but increase memory consumption.
        
        Args:
            dataset_path: Path to styles directory containing JSON files
            batch_size: Number of documents per batch insertion
            
        Returns:
            True if upload successful
        """
        try:
            styles_path = Path(dataset_path)
            if not styles_path.exists():
                logger.error(f"Dataset path does not exist: {dataset_path}")
                return False
            
            # Get all JSON files
            json_files = list(styles_path.glob("*.json"))
            total_files = len(json_files)
            
            if total_files == 0:
                logger.error(f"No JSON files found in {dataset_path}")
                return False
            
            logger.info(f"Found {total_files} JSON files to process")
            
            # Process files in batches
            documents_batch = []
            processed_count = 0
            success_count = 0
            error_count = 0
            
            start_time = time.time()
            
            for i, json_file in enumerate(json_files):
                try:
                    # Read and parse JSON file
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Extract product data
                    document = self.extract_product_data(json_data)
                    if document:
                        documents_batch.append(document)
                        success_count += 1
                    else:
                        error_count += 1
                    
                    processed_count += 1
                    
                    # Insert batch when it reaches batch_size or at the end
                    if len(documents_batch) >= batch_size or i == total_files - 1:
                        if documents_batch:
                            self._insert_batch(documents_batch)
                            logger.info(f"Inserted batch of {len(documents_batch)} documents")
                            documents_batch = []
                    
                    # Progress reporting
                    if processed_count % 1000 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed
                        eta = (total_files - processed_count) / rate if rate > 0 else 0
                        
                        logger.info(
                            f"Progress: {processed_count}/{total_files} "
                            f"({processed_count/total_files*100:.1f}%) "
                            f"Success: {success_count}, Errors: {error_count} "
                            f"Rate: {rate:.1f} files/sec, ETA: {eta:.0f}s"
                        )
                
                except Exception as e:
                    logger.error(f"Error processing file {json_file}: {e}")
                    error_count += 1
                    processed_count += 1
            
            # Final statistics
            elapsed_total = time.time() - start_time
            logger.success(
                f"Dataset upload completed in {elapsed_total:.2f}s\n"
                f"Total files processed: {processed_count}\n"
                f"Successful uploads: {success_count}\n"
                f"Errors: {error_count}\n"
                f"Average rate: {processed_count/elapsed_total:.1f} files/sec"
            )
            
            return error_count == 0
            
        except Exception as e:
            logger.error(f"Dataset upload failed: {e}")
            return False
    
    def _insert_batch(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Insert batch of documents with duplicate handling.
        
        Learning: Bulk operations with ordered=False continue processing
        even if some documents fail, improving overall resilience.
        """
        try:
            # Use bulk write for better performance
            # Learning: unordered bulk operations are faster but may
            # process documents out of order
            result = self.collection.insert_many(
                documents, 
                ordered=False  # Continue on errors
            )
            
            return len(result.inserted_ids) == len(documents)
            
        except errors.BulkWriteError as e:
            # Handle duplicate key errors gracefully
            # Learning: Bulk write errors contain details about which operations failed
            logger.warning(f"Bulk write completed with {len(e.details['writeErrors'])} errors")
            return True  # Consider partial success as success
            
        except Exception as e:
            logger.error(f"Batch insertion failed: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics after upload.
        
        Learning: Collection stats help verify upload success and
        plan vector search index requirements.
        """
        try:
            stats = self.collection.count_documents({})
            sample_doc = self.collection.find_one()
            
            return {
                "total_documents": stats,
                "sample_document": sample_doc,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def close(self):
        """
        Close MongoDB connection.
        
        Learning: Always close database connections to free resources
        and prevent connection pool exhaustion.
        """
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Main execution function
def main():
    """
    Main function to execute dataset upload.
    
    Learning: Environment variables keep sensitive credentials secure
    and allow easy configuration across different environments.
    """
    
    # Configuration
    # Load environment variables from .env file
    load_dotenv()
    MONGODB_URI = os.getenv("MONGODB_URI")
    if not MONGODB_URI:
        logger.error("MONGODB_URI environment variable not set")
        return
    DATABASE_NAME = "fashion_semantic_search"
    COLLECTION_NAME = "products"
    DATASET_PATH = "dataset/styles"
    
    # Initialize uploader
    uploader = FashionDatasetUploader(
        connection_string=MONGODB_URI,
        database_name=DATABASE_NAME,
        collection_name=COLLECTION_NAME
    )
    
    try:
        # Connect to MongoDB
        if not uploader.connect():
            logger.error("Failed to connect to MongoDB - aborting upload")
            return
        
        # Create indexes for optimal query performance
        uploader.create_indexes()
        
        # Upload dataset
        logger.info("Starting dataset upload...")
        success = uploader.upload_dataset(DATASET_PATH, batch_size=500)
        
        if success:
            # Show final statistics
            stats = uploader.get_collection_stats()
            logger.success(f"Upload completed successfully!")
            logger.info(f"Total documents in collection: {stats.get('total_documents', 0)}")
            
            # Next steps guidance
            logger.info(
                "\nNext steps for vector search setup:\n"
                "1. Generate embeddings for the 'description' and 'name' fields\n"
                "2. Create vector search index in MongoDB Atlas UI\n"
                "3. Configure index for 'image_embedding' field (add this field during embedding generation)\n"
                "4. Test semantic search queries\n"
                "\nThe collection is ready for embedding generation!"
            )
        else:
            logger.error("Upload completed with errors - check logs for details")
    
    finally:
        # Always cleanup resources
        uploader.close()

if __name__ == "__main__":
    main()