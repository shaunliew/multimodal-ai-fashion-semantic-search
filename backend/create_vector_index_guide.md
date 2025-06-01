# MongoDB Atlas Vector Search Index Setup Guide

**Learning Focus**: Understanding vector search fundamentals and MongoDB Atlas implementation for semantic fashion search

## What You'll Learn

- **Vector Embeddings**: How neural network embeddings enable semantic similarity search
- **Vector Databases**: Why traditional indexes don't work for high-dimensional vectors  
- **Cosine Similarity**: The mathematical foundation of semantic search
- **MongoDB Atlas Vector Search**: Production-ready vector search implementation

## Technical Overview

### Why Vector Search?

Traditional database indexes work with exact matches or ranges. Vector search enables **semantic similarity** - finding items that are conceptually similar rather than exactly matching.

**Example**: Query "blue denim jacket" finds:

- ✅ "Forever New Women Vintage Blue Jackets" (semantic match)
- ❌ Won't find if no product name contains exact words "blue denim jacket"

**Learning**: CLIP embeddings map images and text into a shared 768-dimensional semantic space where similar concepts cluster together.

## Step-by-Step Setup

### 1. Access MongoDB Atlas Dashboard

1. Navigate to [MongoDB Atlas](https://cloud.mongodb.com/)
2. Log in and select your cluster
3. **Requirement**: M10+ cluster (Vector Search requires Atlas, not Community Edition)

**Learning**: Vector Search uses specialized HNSW (Hierarchical Navigable Small World) algorithms that require Atlas infrastructure.

### 2. Create Vector Search Index

1. **Navigate to Search Indexes**:

   - Database Deployments → Select Cluster → Browse Collections
   - Find database: `fashion_semantic_search`
   - Find collection: `products`
   - Click "Search Indexes" tab

2. **Create New Search Index**:
   - Click "Create Search Index"
   - **Important**: Select "JSON Editor" (not Visual Editor)
   - Choose your database and collection

### 3. Index Configuration

**Copy this exact configuration** (tested and verified):

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

**Learning Breakdown**:

- **`numDimensions: 768`**: Matches CLIP ViT-Large embedding size
- **`path: "image_embedding"`**: Field name containing vector arrays in documents
- **`similarity: "cosine"`**: Measures angle between vectors (ideal for normalized embeddings)
- **`type: "vector"`**: Enables approximate nearest neighbor search

### 4. Index Naming

**Recommended name**: `vector_index` (tested and working)

**Alternative names that work**:

- `image_embedding_index`
- `semantic_search_index`

**Learning**: The test script automatically tries multiple names, but `vector_index` is the proven working configuration.

### 5. Wait for Index Build

- Status progression: "Building" → "Active"
- **Critical**: Wait for "Active" status before testing
- Build time: ~2-5 minutes depending on collection size

**Learning**: HNSW index construction requires preprocessing all vectors to build the navigation graph structure.

## Verification & Testing

### Quick Validation

After index shows "Active", test immediately:

```bash
cd backend
uv run test_vector_search.py
```

**Expected output**:

```bash
✅ Using known working vector index: vector_index
🎉 Vector search is working correctly!
```

### Understanding Your Data Structure

Your MongoDB documents should look like this:

```json
{
  "_id": "product_123",
  "name": "Forever New Women Vintage Blue Jackets",
  "brand": "Forever New", 
  "category": "Jackets",
  "image": "http://assets.myntassets.com/...",
  "image_embedding": [0.123, -0.456, 0.789, ...], // 768 float values
  "description": "Washed light blue biker denim jacket..."
}
```

**Learning**: The `image_embedding` array contains normalized 768-dimensional vectors from CLIP ViT-Large model.

## Advanced Configuration

### Performance Tuning

**For large collections (>100K products)**:

```json
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "image_embedding",
      "similarity": "cosine", 
      "type": "vector"
    }
  ],
  "vectorSearchConfiguration": {
    "numCandidates": 150,
    "limit": 10
  }
}
```

**Learning**: `numCandidates` controls accuracy vs. speed trade-off in HNSW search.

### Search Query Example

Test your index manually in MongoDB Compass:

```javascript
[
  {
    "$vectorSearch": {
      "index": "vector_index",
      "path": "image_embedding", 
      "queryVector": [/* your 768-dimensional query vector */],
      "numCandidates": 50,
      "limit": 5
    }
  },
  {
    "$addFields": {
      "similarity_score": { "$meta": "vectorSearchScore" }
    }
  }
]
```

**Learning**: `$vectorSearch` performs approximate nearest neighbor search, `$meta: "vectorSearchScore"` returns cosine similarity.

## Troubleshooting

### Common Issues & Solutions

#### ❌ "No vector search index found"

**Cause**: Index not created or wrong name
**Solution**: Verify index name is `vector_index` and status is "Active"

#### ❌ "Wrong dimensions"

**Cause**: Mismatch between embedding size and index configuration
**Solution**: Verify embeddings are 768-dimensional:

```bash
# Check in MongoDB Compass
db.products.findOne({"image_embedding": {"$exists": true}}, {"image_embedding": 1}).image_embedding.length
// Should return: 768
```

#### ❌ "Index building failed"

**Cause**: Invalid embedding data (null, wrong type, inconsistent dimensions)
**Solution**: Validate all embeddings before index creation:

```bash
# Count products with valid embeddings
db.products.countDocuments({
  "image_embedding": {"$type": "array", "$size": 768}
})
```

#### ⚠️ "Poor search results"

**Cause**: Non-normalized embeddings or quality issues
**Solution**: Check embedding statistics - mean norm should be ~1.0:

```python
# The test script shows: "mean_norm": 1.0, "expected_norm": "~1.0 (normalized embeddings)"
```

### Performance Diagnostics

**Expected search performance**:

- Excellent: <0.5 seconds
- Good: <1.0 seconds
- Needs improvement: >1.0 seconds

**Learning**: Vector search performance depends on:

1. Collection size (logarithmic scaling with HNSW)
2. Network latency to Atlas
3. Index freshness and optimization

## Next Steps

### 1. ✅ Verify Setup

```bash
uv run test_vector_search.py
```

### 2. 🧪 Experiment with Queries

```python
# Test different semantic concepts
queries = [
    "blue denim jacket",      # Color + material + type
    "formal business attire", # Context-based search
    "summer casual wear",     # Season + style
    "elegant evening dress"   # Mood + occasion
]
```

### 3. 🚀 Integrate with Frontend

- Build search interface with real-time results
- Add image upload for visual similarity search
- Implement search filters and result ranking

### 4. 📊 Monitor Performance

- Track search response times
- Analyze query patterns and popular searches
- Monitor index utilization and optimization needs

## Learning Resources

- [MongoDB Vector Search Documentation](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [CLIP Paper: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [HNSW Algorithm Explained](https://arxiv.org/abs/1603.09320)
- [Vector Database Fundamentals](https://www.pinecone.io/learn/vector-database/)

---

**Success Criteria**: When `test_vector_search.py` shows all tests passing with good similarity scores (>0.6) and fast response times (<1.0s), your vector search is production-ready for semantic fashion discovery.
