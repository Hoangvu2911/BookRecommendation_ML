# Book Recommendation System

A production-ready Flask API combining **Content-Based**, **KNN Collaborative Filtering**, and **SVD Matrix Factorization** for personalized book recommendations.

## Features

- **Combined Recommendation Engine**: 3 models (weights: content-0.2 + KNN-0.3 + SVD-0.5)
- **Full-Text Search**: SQLite FTS for fast book search
- **Real-time Recommendations**: Per-user personalized suggestions
- **Modular Architecture**: Clean separation of concerns
- **Auto-initialization**: Setup scripts run automatically on first start
- **Production Ready**: Logging, error handling, graceful shutdown

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Server

```bash
python app.py
```

Server automatically:

- Checks/generates models
- Checks/creates database
- Loads all models into RAM
- Starts Flask server on `http://localhost:5000`