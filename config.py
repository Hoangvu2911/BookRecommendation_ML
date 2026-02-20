#!/usr/bin/env python3
"""
Configuration file for Book Recommendation System
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Database configuration
DATABASE_PATH = str(PROJECT_ROOT / 'books_data.db')

# Data files configuration
DATA_DIR = PROJECT_ROOT / 'data'
CSV_FILES = {
    'books': str(DATA_DIR / 'book.csv'),
    'ratings': str(DATA_DIR / 'rating.csv'),
}

# Models configuration
MODELS_DIR = PROJECT_ROOT / 'pkl'
MODEL_FILES = {
    'cosine_sim': 'cosine_sim.pkl',
    'knn_model': 'knn_model.pkl',
    'svd_model': 'svd_model.pkl',
}

MODEL_FILE_PATHS = {key: str(MODELS_DIR / value) for key, value in MODEL_FILES.items()}

# Recommendation configuration
RECOMMENDATION_CONFIG = {
    'weights': {
        'content_based': 0.2,  # TF-IDF + Cosine Similarity
        'knn': 0.3,            # Item-Item Collaborative Filtering
        'svd': 0.5             # Matrix Factorization
    },
    'default_top_n': 10,
    'max_top_n': 100,
    'rating_scale': (1, 10)
}

# Flask configuration
FLASK_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'use_reloader': False,
    'threaded': True
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(PROJECT_ROOT / 'app.log')
}

# Scripts configuration
SCRIPTS_CONFIG = {
    'export_models_timeout': 600,
    'create_database_timeout': 300
}
