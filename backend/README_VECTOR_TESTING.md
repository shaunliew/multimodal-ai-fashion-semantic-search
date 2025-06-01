# Vector Search Testing Guide

This guide explains how to test your MongoDB Atlas Vector Search implementation for fashion image embeddings.

## Overview

The `test_vector_search.py` script validates that your CLIP-based image embeddings are working correctly for semantic fashion search. It performs comprehensive testing including:

- **Infrastructure validation**: Checks MongoDB Atlas vector index configuration
- **Data quality analysis**: Validates embedding dimensions and normalization
- **Text-to-image search**: Tests cross-modal semantic search capabilities
- **Image-to-image search**: Tests visual similarity search
- **Performance benchmarking**: Measures search response times

## Prerequisites

### 1. MongoDB Atlas Vector Search Index

You must have created a vector search index in MongoDB Atlas with these specifications:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "image_embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    }
  ]
}
```

**Index Name**: `image_embedding_index` (or update the script if different)

### 2. Environment Setup

Create a `.env` file in the backend directory:

```bash
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
```

### 3. Install Dependencies

```bash
pip install -r requirements-test.txt
```

## Running the Tests

### Basic Test Run

```bash
cd backend
python test_vector_search.py
```

This will run the complete test suite and generate a detailed report.

### Expected Output

```bash
🚀 Starting Vector Search Validation Tests...

📋 Phase 1: Infrastructure Validation
🔍 Validating vector search index configuration...
✅ Found vector index: image_embedding_index
📊 Validating embedding data quality...
✅ Embedding quality looks good!

🔍 Phase 2: Search Functionality Tests
🔤 Testing text-to-image search: 'blue denim jacket'
✅ Found 3 similar products
   Similarity range: 0.756 - 0.892

🖼️ Testing image-to-image similarity search
✅ Found 3 visually similar products

⚡ Phase 3: Performance Analysis

📊 Phase 4: Overall Assessment
🎉 Vector search is working correctly!
```

## Understanding Test Results

### Test Report Sections

1. **Infrastructure Tests**
   - ✅ Vector Index: Confirms index exists and is configured correctly
   - ✅ Embeddings: Shows data coverage and quality metrics

2. **Functionality Tests**
   - Text-to-Image Search: Tests semantic search with natural language
   - Image-to-Image Search: Tests visual similarity matching

3. **Performance**
   - Average Response Time: Typical search latency
   - Performance Rating: excellent (<0.5s), good (<1.0s), needs_improvement (>1.0s)

### Success Criteria

- **PASS**: All infrastructure OK, most searches return results
- **PARTIAL**: Infrastructure OK, but limited search results
- **FAIL**: Missing index or embeddings, searches failing

## Test Query Examples

The script tests with various fashion queries:

```python
test_queries = [
    "blue denim jacket",
    "red summer dress", 
    "black leather boots",
    "white cotton t-shirt",
    "elegant evening gown"
]
```

You can modify these in the script for your specific use case.

## Troubleshooting

### Common Issues

#### ❌ No vector search index found

**Problem**: Vector index doesn't exist in MongoDB Atlas

**Solution**:

1. Go to MongoDB Atlas → Database → Search → Create Index
2. Select "Vector Search"
3. Use configuration shown in Prerequisites section

#### ❌ No embeddings found in collection

**Problem**: Products don't have `image_embedding` field

**Solution**: Run the embedding generation script first:

```bash
python generate_embeddings.py
```

#### ❌ Index validation failed: list_search_indexes not supported

**Problem**: Using MongoDB Community Edition (not Atlas)

**Solution**: Vector Search requires MongoDB Atlas M10+ cluster

#### ⚠️ No results found

**Problem**: Search queries not matching any products

**Possible causes**:

- Embedding quality issues
- Similarity threshold too high
- Limited product diversity in dataset

**Solutions**:

- Lower similarity threshold: `similarity_threshold=0.3`
- Check embedding normalization (mean norm should be ~1.0)
- Try more general queries like "shirt" or "dress"

### Performance Issues

#### Slow search responses (>2 seconds)

**Causes**:

- Large collection without proper indexing
- Network latency to MongoDB Atlas
- CLIP model loading on CPU

**Solutions**:

- Ensure vector index is properly created
- Use MongoDB Atlas in same region
- Run tests on GPU-enabled machine for faster embeddings

## Advanced Testing

### Custom Test Queries

Modify the test script to use your own queries:

```python
# In test_vector_search.py, line ~610
custom_queries = [
    "your custom query 1",
    "your custom query 2"
]
```

### Image-to-Image Testing with External Images

Test with your own reference image:

```python
result = tester.test_image_to_image_search(
    reference_image_url="https://example.com/your-image.jpg",
    limit=5
)
```

### Similarity Threshold Tuning

Test different similarity thresholds:

```python
for threshold in [0.3, 0.5, 0.7, 0.9]:
    result = tester.test_text_to_image_search(
        "blue shirt", 
        similarity_threshold=threshold
    )
```

## Next Steps

Once your tests pass:

1. **Frontend Integration**: Build a web interface for search
2. **API Development**: Create FastAPI endpoints for search functionality
3. **Production Optimization**: Implement caching and rate limiting
4. **User Experience**: Add filters, categories, and search refinements

## Support

If you encounter issues:

1. Check the generated `test_results_[timestamp].json` for detailed diagnostics
2. Review MongoDB Atlas logs for index-related issues
3. Verify your embedding generation completed successfully
4. Ensure your MongoDB Atlas cluster is M10+ (required for Vector Search)

**Learning Resources**:

- [MongoDB Atlas Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)

- [CLIP Model Paper](https://arxiv.org/abs/2103.00020)

- [Fashion Semantic Search Best Practices](https://github.com/mongodb-developer/image-search-vector-demo)
