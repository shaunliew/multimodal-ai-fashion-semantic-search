# Backend API

A FastAPI-based backend service.

## Setup

1. Ensure you have Python 3.10+ installed
2. Install uv package manager:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create and activate the environment:

   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # OR
   .venv\Scripts\activate  # On Windows
   ```

4. Install dependencies:

   ```bash
   uv sync
   ```

## Running the Server

Start the development server:

```bash
uvicorn main:app --reload
```

The API will be available at <http://127.0.0.1:8000>

API documentation is available at:

- <http://127.0.0.1:8000/docs>
- <http://127.0.0.1:8000/redoc>
