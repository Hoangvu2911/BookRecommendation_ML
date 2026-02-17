#!/usr/bin/env python3
"""
AI Model management for Book Recommendation System
"""

import joblib
import pandas as pd
from config import MODEL_FILE_PATHS, CSV_FILES
from utils.logger import setup_logger

logger = setup_logger(__name__)

class ModelManager:
    """
    Central manager for loading and managing all ML models
    """
    
    def __init__(self):
        """Initialize model manager"""
        self.models = {}
        self.data = {}
        self.loaded = False
    
    def load_all_models(self):
        """
        Load all required models and data
        
        Returns:
            bool: True if all models loaded successfully
        """
        logger.info("🧠 Loading all models and data...")
        
        try:
            # Load model files
            self.models['tfidf_vectorizer'] = joblib.load(MODEL_FILE_PATHS['tfidf_vectorizer'])
            logger.info("✓ Loaded TF-IDF Vectorizer")
            
            self.models['tfidf_matrix'] = joblib.load(MODEL_FILE_PATHS['tfidf_matrix'])
            logger.info(f"✓ Loaded TF-IDF Matrix: {self.models['tfidf_matrix'].shape}")
            
            self.models['isbn_map'] = joblib.load(MODEL_FILE_PATHS['isbn_map'])
            logger.info(f"✓ Loaded ISBN Map: {len(self.models['isbn_map']['isbn_to_index'])} books")
            
            self.models['cosine_sim'] = joblib.load(MODEL_FILE_PATHS['cosine_sim'])
            logger.info(f"✓ Loaded Cosine Similarity Matrix: {self.models['cosine_sim'].shape}")
            
            self.models['knn_model'] = joblib.load(MODEL_FILE_PATHS['knn_model'])
            logger.info("✓ Loaded KNN Model")
            
            self.models['svd_model'] = joblib.load(MODEL_FILE_PATHS['svd_model'])
            logger.info("✓ Loaded SVD Model")
            
            # Load data files
            logger.info("📚 Loading data files...")
            self.data['books'] = pd.read_csv(
                CSV_FILES['books'],
                usecols=['ISBN', 'Title', 'Author', 'description', 'genres']
            )
            logger.info(f"✓ Loaded {len(self.data['books'])} books")
            
            self.data['ratings'] = pd.read_csv(CSV_FILES['ratings'])
            logger.info(f"✓ Loaded {len(self.data['ratings'])} ratings")
            
            self.loaded = True
            logger.info("✓ All models and data loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            self.loaded = False
            return False
    
    def get_model(self, name):
        """
        Get a loaded model by name
        
        Args:
            name: Model name
            
        Returns:
            Model object or None if not found
        """
        return self.models.get(name)
    
    def get_data(self, name):
        """
        Get loaded data by name
        
        Args:
            name: Data name (e.g., 'books', 'ratings')
            
        Returns:
            DataFrame or None if not found
        """
        return self.data.get(name)
    
    def is_loaded(self):
        """
        Check if all models are loaded
        
        Returns:
            bool: True if models are loaded
        """
        return self.loaded
    
    def get_status(self):
        """
        Get status of loaded models and data
        
        Returns:
            dict: Status information
        """
        return {
            'loaded': self.loaded,
            'models': list(self.models.keys()),
            'data': list(self.data.keys()),
            'model_count': len(self.models),
            'data_count': len(self.data)
        }

# Global model manager instance
_model_manager = None

def get_model_manager():
    """
    Get or create the global model manager instance
    
    Returns:
        ModelManager: Global model manager
    """
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

def ensure_models_loaded():
    """
    Ensure models are loaded, raise error if not
    
    Returns:
        ModelManager: The model manager
        
    Raises:
        RuntimeError: If models are not loaded
    """
    manager = get_model_manager()
    if not manager.is_loaded():
        raise RuntimeError("Models are not loaded. Call load_all_models() first.")
    return manager
