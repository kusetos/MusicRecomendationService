# MusicRecomendationService

This is a standalone FastAPI microservice that provides song recommendations based on precomputed embeddings.

## Features
- **GET /recommendations/{user_id}?limit={n}**: Returns top-N recommended song IDs for a given user.

## Requirements
- Python 3.9+
- pip

## Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows

# Install dependencies
pip install fastapi uvicorn scikit-learn numpy
```

## Running
```bash
uvicorn main:app --reload
```

The server starts at http://127.0.0.1:8000

## Usage Example
```
GET http://127.0.0.1:8000/recommendations/1?limit=5
```

