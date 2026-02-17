#!/usr/bin/env python3
"""
AI module for Book Recommendation System
"""

from .models import ModelManager, get_model_manager, ensure_models_loaded
from .recommender import Recommender

__all__ = [
    'ModelManager',
    'get_model_manager',
    'ensure_models_loaded',
    'Recommender'
]
