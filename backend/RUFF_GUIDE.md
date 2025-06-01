# Ruff Configuration Guide for Semantic Fashion Search Project

## What is Ruff?

Ruff is an extremely fast Python linter and formatter written in Rust. In this semantic fashion search project, Ruff helps maintain code quality by enforcing:

- **Google-style docstrings** for educational clarity
- **Comprehensive type hints** for AI/ML operations
- **Organized imports** for better readability
- **Modern Python patterns** for async FastAPI code

## Why Ruff for This Project?

- **Speed**: 10-100x faster than traditional Python linters
- **Educational Focus**: Enforces documentation standards that support learning
- **AI/ML Optimized**: Catches common issues in async ML workflows
- **Modern Python**: Supports latest features used in FastAPI and vector databases

## Project-Specific Configuration

Our `ruff.toml` is tailored for this semantic search backend:

```toml
target-version = "py310"
line-length = 100

# Educational and AI/ML focused rules
lint.select = [
    "D",     # pydocstyle (Google-style docstrings)
    "ANN",   # Type annotations for ML functions
    "ASYNC", # Async/await patterns for FastAPI
    # ... plus standard quality rules
]

# Allow educational code patterns
lint.ignore = [
    "ERA001",  # Allow educational comments
    "D107",    # Class docstrings can cover __init__
]
```

## Code Style Examples from main.py

### 1. Google-Style Docstrings (Required)

**Good** - Educational docstring from main.py:
```python
async def generate_text_embedding(text: str) -> List[float]:
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
```

**Bad** - Missing or incomplete docstring:
```python
def generate_text_embedding(text: str) -> List[float]:
    # Convert text to embedding
    return embedding
```

### 2. Type Hints (Enforced)

**Good** - Comprehensive type hints from main.py:
```python
async def perform_vector_search(
    query_embedding: List[float],
    limit: int,
    similarity_threshold: float = 0.0,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
```

**Bad** - Missing type annotations:
```python
async def perform_vector_search(query_embedding, limit, similarity_threshold=0.0):
    return results
```

### 3. Import Organization

**Good** - Grouped imports as in main.py:
```python
# Standard library
import os
import time
from typing import List, Dict, Any

# Third-party
import torch
from fastapi import FastAPI
from pydantic import BaseModel

# Local imports
from app.models import schemas
```

**Bad** - Unorganized imports:
```python
from fastapi import FastAPI
import os
from app.models import schemas
import torch
```

### 4. Educational Comments (Allowed)

Our configuration allows extensive educational comments:
```python
# Learning: MongoDB's $vectorSearch uses approximate nearest neighbor
# algorithms for efficient similarity search in high-dimensional space
pipeline = [
    {
        "$vectorSearch": {
            "index": resources.working_index_name,
            # ... configuration
        }
    }
]
```

## Using Ruff in Development

### Daily Commands

Check your code (run this before commits):
```bash
cd backend
uv run ruff check main.py
```

Auto-fix issues where possible:
```bash
uv run ruff check --fix main.py
```

Format code to match project style:
```bash
uv run ruff format main.py
```

## Common Issues Ruff Catches

### 1. Missing Docstrings for AI Functions

```python
# ❌ Will be flagged
async def generate_image_embedding(image):
    return embedding

# ✅ Correct
async def generate_image_embedding(image: Image.Image) -> List[float]:
    """
    Generate CLIP embedding for image.
    
    Learning: Image embeddings capture visual features like
    color, pattern, style, and composition for similarity search.
    """
```

### 2. Incomplete Type Annotations
```python
# ❌ Missing return type
async def search_by_text(request: TextSearchRequest):
    return results

# ✅ Complete annotations
async def search_by_text(request: TextSearchRequest) -> SearchResponse:
    return results
```

### 3. Import Order Issues

```python
# ❌ Wrong order
from pydantic import BaseModel
import os
from fastapi import FastAPI

# ✅ Properly grouped
import os

from fastapi import FastAPI
from pydantic import BaseModel
```

### 4. Line Length Violations
```python
# ❌ Too long (>100 chars)
very_long_variable_name = some_function_with_many_parameters(param1, param2, param3, param4, param5)

# ✅ Properly formatted
very_long_variable_name = some_function_with_many_parameters(
    param1, param2, param3, param4, param5
)
```

## AI/ML Specific Checks

Ruff helps catch common issues in ML code:

### Async Pattern Validation

```python
# ✅ Proper async pattern from main.py
async def generate_text_embedding(text: str) -> List[float]:
    with torch.no_grad():
        # AI model operations
        pass
```

### Vector Type Safety
```python
# ✅ Clear vector typing
query_embedding: List[float]
similarity_scores: Dict[str, float]
```

## Configuration for Learning

Our setup prioritizes educational value:

- **Longer variable names allowed** for clarity
- **Educational comments preserved** 
- **Comprehensive docstrings required** for all public functions
- **Type hints mandatory** to understand AI operations

## Pre-commit Integration

Add to `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

This ensures all commits follow our educational coding standards and help maintain the learning-focused quality of the semantic search project.

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

Remember: Every lint rule in this project serves the dual purpose of code quality and educational clarity for learning AI/ML concepts!
