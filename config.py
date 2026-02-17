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
DATABASE_TABLES = {
    'books': 'Main book metadata with full-text search',
    'books_fts': 'Full-text search index',
    'top5': 'Top 5 rated books'
}

# Data files configuration
DATA_DIR = PROJECT_ROOT / 'data'
CSV_FILES = {
    'books': str(DATA_DIR / 'book.csv'),
    'ratings': str(DATA_DIR / 'rating.csv'),
    'des_cross': str(DATA_DIR / 'des_cross.csv'),
    'des_book': str(DATA_DIR / 'des_book.csv'),
}

# Models configuration
MODELS_DIR = PROJECT_ROOT
MODEL_FILES = {
    'tfidf_vectorizer': 'tfidf_vectorizer.pkl',
    'tfidf_matrix': 'tfidf_matrix.pkl',
    'isbn_map': 'isbn_map.pkl',
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

# TF-IDF configuration
TFIDF_CONFIG = {
    'stop_words': 'english',
    'min_df': 2,
    'max_features': None
}

# KNN configuration
KNN_CONFIG = {
    'k_range': (10, 50),
    'sim_metric': 'pearson',
    'user_based': False
}

# SVD configuration
SVD_CONFIG = {
    'factor_candidates': [2, 3, 4, 5, 10, 15, 20, 50, 75, 100, 150, 200],
    'n_epochs': 30,
    'lr_all': 1e-3,
    'reg_all': 0.05,
    'random_state': 42
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
