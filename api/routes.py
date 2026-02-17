#!/usr/bin/env python3
"""
API Routes for Book Recommendation System
"""

from flask import request, jsonify, render_template, send_file
import os
from config import RECOMMENDATION_CONFIG
from utils.logger import setup_logger
from utils.database import (
    get_top5_books, 
    search_books, 
    get_book_by_isbn,
    check_database_exists,
    enrich_recommendations_with_metadata
)
from ai import get_model_manager, Recommender

logger = setup_logger(__name__)

def register_routes(app):
    """
    Register all API routes
    
    Args:
        app: Flask application instance
    """
    
    # Get the root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    @app.route('/', methods=['GET'])
    def index():
        """Serve the main index.html page"""
        return render_template('index.html')
    
    @app.route('/css/<path:filename>', methods=['GET'])
    def serve_css(filename):
        """Serve CSS files"""
        css_dir = os.path.join(root_dir, 'css')
        return send_file(os.path.join(css_dir, filename), mimetype='text/css')
    
    @app.route('/js/<path:filename>', methods=['GET'])
    def serve_js(filename):
        """Serve JavaScript files"""
        js_dir = os.path.join(root_dir, 'js')
        return send_file(os.path.join(js_dir, filename), mimetype='application/javascript')
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        try:
            model_manager = get_model_manager()
            
            status = {
                'status': 'healthy',
                'database': {
                    'connected': check_database_exists()
                },
                'models': {
                    'loaded': model_manager.is_loaded(),
                    'details': model_manager.get_status()
                },
                'recommendation': {
                    'strategy': 'combined (content-0.2 + knn-0.3 + svd-0.5)',
                    'weights': RECOMMENDATION_CONFIG['weights']
                }
            }
            return jsonify(status), 200
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return jsonify({'status': 'error', 'error': str(e)}), 500
    
    @app.route('/api/top5', methods=['GET'])
    def get_top5():
        """Get top 5 rated books"""
        try:
            books = get_top5_books()
            
            response = {
                'success': True,
                'count': len(books),
                'books': books
            }
            
            logger.info(f"Top5 API: Returned {len(books)} books")
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Top5 API error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/search', methods=['GET'])
    def search():
        """Search for books using full-text search"""
        try:
            query = request.args.get('q', '').strip()
            limit = int(request.args.get('limit', 20))
            
            if not query:
                return jsonify({
                    'success': False,
                    'error': 'Query parameter "q" is required'
                }), 400
            
            if len(query) < 2:
                return jsonify({
                    'success': False,
                    'error': 'Query must be at least 2 characters'
                }), 400
            
            results = search_books(query, limit=limit)
            
            response = {
                'success': True,
                'query': query,
                'count': len(results),
                'results': results
            }
            
            logger.info(f"Search API: Query '{query}' returned {len(results)} results")
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Search API error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/book/<isbn>', methods=['GET'])
    def get_book(isbn):
        """Get book details with TF-IDF recommendations"""
        try:
            book = get_book_by_isbn(isbn)
            
            if not book:
                return jsonify({
                    'success': False,
                    'error': 'Book not found'
                }), 404
            
            # Get TF-IDF recommendations if models are loaded
            recommendations = []
            if get_model_manager().is_loaded():
                # TODO: Implement TF-IDF recommendation computation if needed
                logger.info(f"TF-IDF recommendations for ISBN {isbn} (not yet implemented in modular structure)")
            
            response = {
                'success': True,
                'book': book,
                'recommendations': recommendations
            }
            
            logger.info(f"Book details API: ISBN {isbn}")
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Book details API error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/api/user/<int:user_id>/recommendations', methods=['GET'])
    def get_user_recommendations(user_id):
        """Get personalized recommendations for user"""
        try:
            model_manager = get_model_manager()
            
            if not model_manager.is_loaded():
                return jsonify({
                    'success': False,
                    'error': 'Models not loaded'
                }), 503
            
            # Get parameters
            try:
                top_n = int(request.args.get('top_n', RECOMMENDATION_CONFIG['default_top_n']))
                if top_n < 1 or top_n > RECOMMENDATION_CONFIG['max_top_n']:
                    top_n = RECOMMENDATION_CONFIG['default_top_n']
            except:
                top_n = RECOMMENDATION_CONFIG['default_top_n']
            
            # Initialize recommender
            book_df = model_manager.get_data('books')
            rating_df = model_manager.get_data('ratings')
            
            recommender = Recommender(book_df, rating_df)
            
            # Get recommendations
            logger.info(f"Computing recommendations for User {user_id}...")
            recs = recommender.recommend_for_user(user_id, n=top_n)
            
            if not recs:
                return jsonify({
                    'success': True,
                    'user_id': user_id,
                    'strategy': 'combined (content + knn + svd)',
                    'note': 'No recommendations available',
                    'recommendations': []
                }), 200
            
            # Enrich with database metadata
            enriched_recs = enrich_recommendations_with_metadata(recs)
            
            # User stats
            user_ratings = rating_df[rating_df['User-ID'] == user_id]
            
            response = {
                'success': True,
                'user_id': user_id,
                'user_stats': {
                    'total_ratings': len(user_ratings),
                    'avg_rating': float(user_ratings['Rating'].mean()) if len(user_ratings) > 0 else 0
                },
                'strategy': 'combined (content-0.2 + knn-0.3 + svd-0.5)',
                'recommendation_count': len(enriched_recs),
                'recommendations': enriched_recs
            }
            
            logger.info(f"✓ User {user_id}: {len(enriched_recs)} recommendations")
            return jsonify(response), 200
            
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Invalid user_id format (must be integer)'
            }), 400
        except Exception as e:
            logger.error(f"User recommendations API error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
