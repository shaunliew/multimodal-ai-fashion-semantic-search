# Multimodal AI Fashion Semantic Search

A semantic search engine for fashion items that allows searching using both text and images.

## Features

- Text-based semantic search for fashion items
- Image-based search functionality
- Responsive UI built with Next.js
- RESTful API with FastAPI
- Data storage with MongoDB Atlas

## Tech Stack

- **Frontend**: Next.js
- **Backend**: FastAPI
- **Database**: MongoDB Atlas

## Setup

### Prerequisites

- Node.js
- Python 3.8+
- UV (Python package manager)
- MongoDB Atlas account

### Installation

1. Clone the repository
2. Install frontend dependencies:

   ```bash
   cd frontend
   npm install
   ```

3. Install backend dependencies:

   ```bash
   cd backend
   uv pip install -r requirements.txt
   ```

4. Configure MongoDB Atlas connection string in the backend environment variables

## Usage

1. Start the backend server:

   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. Start the frontend development server:

   ```bash
   cd frontend
   npm run dev
   ```

3. Access the application at `http://localhost:3000`

feature

- image search
- text search
