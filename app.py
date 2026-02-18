#!/usr/bin/env python3
"""
Book Recommendation System - Flask Backend Server

Architecture:
- Modular design with separated concerns
- Config: config.py (centralized configuration)
- Utils: database, logging, setup helpers
- AI: model management and loading
- API: routes and endpoints
- Core: recommender.py (recommendation logic)

Setup:
1. Download pre-trained models from Google Drive and place in project root:
   - tfidf_vectorizer.pkl
   - tfidf_matrix.pkl
   - cosine_sim.pkl
   - knn_model.pkl
   - svd_model.pkl
   - isbn_map.pkl

2. Run create_database.py to initialize the database

Usage:
    python app.py

The server will automatically:
1. Check if models exist (must be downloaded from Google Drive)
2. Check if database exists, run create_database.py if needed
3. Load all models and data into memory
4. Start Flask server on http://localhost:5000
"""

import signal
import sys
import os
from flask import Flask
from config import FLASK_CONFIG
from utils.logger import setup_logger
from utils.setup import check_and_setup
from ai.models import get_model_manager
from api.routes import register_routes

# Setup logger
logger = setup_logger(__name__)

def create_app():
    """
    Create and configure Flask application
    
    Returns:
        Flask: Configured Flask app instance
    """
    # Get the root directory of the project
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create Flask app with template folder pointing to root directory
    app = Flask(__name__, template_folder=root_dir, static_folder=os.path.join(root_dir, 'css'), static_url_path='/css')
    
    # Register API routes
    register_routes(app)
    
    # CORS headers
    @app.after_request
    def add_cors_headers(response):
        """Add CORS headers to all responses"""
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    return app

def main():
    """Main entry point"""
    logger.info("=" * 80)
    logger.info("BOOK RECOMMENDATION SYSTEM - FLASK SERVER")
    logger.info("=" * 80)
    
    try:
        # Step 1: Initialize (check and setup models/database)
        logger.info("\n📋 STEP 1: System Initialization")
        check_and_setup()
        
        # Step 2: Load models
        logger.info("\n🤖 STEP 2: Loading AI Models")
        model_manager = get_model_manager()
        if not model_manager.load_all_models():
            raise RuntimeError("Failed to load models")
        
        # Step 3: Create Flask app
        logger.info("\n🚀 STEP 3: Starting Flask Server")
        app = create_app()
        
        # Step 4: Print server info
        logger.info("\n" + "=" * 80)
        logger.info("✓ SERVER READY!")
        logger.info("=" * 80)
        logger.info(f"\nServer: http://{FLASK_CONFIG['host']}:{FLASK_CONFIG['port']}")
        logger.info("\nAPI Endpoints:")
        logger.info("  GET  /api/health                          - Health check")
        logger.info("  GET  /api/top5                            - Get top 5 books")
        logger.info("  GET  /api/search?q=<query>                - Search books")
        logger.info("  GET  /api/book/<isbn>                     - Get book details")
        logger.info("  GET  /api/user/<user_id>/recommendations  - Get personalized recommendations")
        logger.info("\nRecommendation Strategy:")
        logger.info("  Weights: Content-Based 0.2 | KNN 0.3 | SVD 0.5")
        logger.info("\nPress Ctrl+C to stop server")
        logger.info("=" * 80 + "\n")
        
        # Step 5: Start server
        app.run(
            host=FLASK_CONFIG['host'],
            port=FLASK_CONFIG['port'],
            debug=FLASK_CONFIG['debug'],
            use_reloader=FLASK_CONFIG['use_reloader'],
            threaded=FLASK_CONFIG['threaded']
        )
        
    except Exception as e:
        logger.error(f"❌ FATAL ERROR: {e}")
        logger.error("Server startup failed")
        sys.exit(1)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("\n" + "=" * 80)
    logger.info("⏹️  Shutting down server...")
    logger.info("=" * 80)
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
