#!/usr/bin/env python3
"""
FLASK BACKEND API SERVER
Auto-setup: Automatically runs export_models.py and create_database.py if needed
Load đỉnh server: Load 4 file .pkl + rating data vào RAM
Load lúc request: Query database, tính recommendation trực tiếp

API Endpoints:
1. /api/top5 - Get top 5 books (from DB)
2. /api/search - Search for books (from DB FTS)
3. /api/book/<isbn> - Get book details with TF-IDF recommendations
4. /api/user/<user_id>/recommendations - USER-BASED recommendations [NEW]

Files:
- export_models.py: Exports TF-IDF + KNN models to .pkl
- create_database.py: Creates SQLite database with FTS
- app.py: Flask server (runs above if needed)
"""

from flask import Flask, request, jsonify, send_file
import sqlite3
import logging
from typing import List, Dict, Any
import os
import joblib
import subprocess
import signal
import sys
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
DATABASE = 'books_data.db'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# SETUP: Auto-run model export & database creation if needed
# ============================================================

def check_and_setup():
    """Kiểm tra xem models và database đã tồn tại chưa, nếu không thì tự động tạo"""
    
    required_models = [
        'tfidf_vectorizer.pkl',
        'tfidf_matrix.pkl',
        'isbn_map.pkl',
        'knn_model.pkl'
    ]
    
    models_exist = all(os.path.exists(f) for f in required_models)
    database_exists = os.path.exists(DATABASE)
    
    if not models_exist:
        logger.info("⚠️ Models not found. Running export_models.py...")
        try:
            result = subprocess.run(['python3', 'export_models.py'], 
                                  capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                logger.info("✓ Models exported successfully!")
            else:
                logger.error(f"❌ Model export failed:\n{result.stderr}")
                raise RuntimeError("Failed to export models")
        except Exception as e:
            logger.error(f"❌ Error exporting models: {e}")
            raise
    
    if not database_exists:
        logger.info("⚠️ Database not found. Running create_database.py...")
        try:
            result = subprocess.run(['python3', 'create_database.py'],
                                  capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info("✓ Database created successfully!")
            else:
                logger.error(f"❌ Database creation failed:\n{result.stderr}")
                raise RuntimeError("Failed to create database")
        except Exception as e:
            logger.error(f"❌ Error creating database: {e}")
            raise

# Run setup check before loading models
check_and_setup()

# ============================================================
# STARTUP: Load Models + Rating Data into RAM
# ============================================================

logger.info("🧠 LOADING MODELS & DATA INTO RAM...")

try:
    # Load TF-IDF components
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    logger.info(f"✓ Loaded TF-IDF Matrix: {tfidf_matrix.shape}")
    
    # Load ISBN index map
    isbn_map = joblib.load('isbn_map.pkl')
    isbn_to_index = isbn_map['isbn_to_index']
    index_to_isbn = isbn_map['index_to_isbn']
    logger.info(f"✓ Loaded ISBN Map: {len(isbn_to_index)} books")
    
    # Load KNN model (Collaborative Filtering)
    knn_model = joblib.load('knn_model.pkl')
    logger.info("✓ Loaded KNN Model")
    
    MODELS_LOADED = True
    
except Exception as e:
    logger.error(f"❌ ERROR loading models: {str(e)}")
    logger.warning("⚠️  Running without models - only top5 and search available")
    MODELS_LOADED = False
    tfidf_matrix = None
    isbn_to_index = None
    index_to_isbn = None
    knn_model = None

# Load Rating Data for User-Based Recommendations
logger.info("👤 Loading rating data for user recommendations...")
try:
    rating_df = pd.read_csv('data/rating.csv')
    logger.info(f"✓ Loaded {len(rating_df)} ratings from {rating_df['User-ID'].nunique()} users")
    RATINGS_LOADED = True
except Exception as e:
    logger.error(f"❌ ERROR loading ratings: {str(e)}")
    RATINGS_LOADED = False
    rating_df = None

# ============================================================
# Helper: TF-IDF Real-time Recommendation
# ============================================================

def get_tf_idf_recommendations(isbn, top_n=5):
    """
    Tính real-time recommendations dùng TF-IDF cosine similarity
    Bước 1: Dò ISBN trong bản đồ
    Bước 2: Lấy vector từ ma trận TF-IDF
    Bước 3: Tính cosine similarity với tất cả sách
    Bước 4: Lấy top 5 tương tự
    """
    if not MODELS_LOADED or isbn not in isbn_to_index:
        return []
    
    try:
        # Bước 1: Tìm index của sách
        book_index = isbn_to_index[isbn]
        
        # Bước 2-3: Tính cosine similarity
        book_vector = tfidf_matrix[book_index]
        similarities = cosine_similarity(book_vector, tfidf_matrix)[0]
        
        # Bước 4: Lấy top 5 (bỏ qua chính nó)
        top_indices = np.argsort(similarities)[::-1][1:top_n+1]
        recommended_isbns = [index_to_isbn[idx] for idx in top_indices]
        
        logger.debug(f"TF-IDF: Book {isbn} -> {len(recommended_isbns)} recommendations")
        return recommended_isbns
        
    except Exception as e:
        logger.error(f"Error calculating TF-IDF recommendations: {str(e)}")
        return []

# ============================================================
# Helper: User-Based Recommendations (dựa trên rating history)
# ============================================================

def get_user_recommendations(user_id, top_n=5, rating_threshold=4):
    """
    Lấy recommendations dựa trên rating history của user
    
    Quy trình:
    1. Lấy tất cả books mà user rate cao (>= rating_threshold)
    2. Dùng TF-IDF tìm similar books cho mỗi quyển yêu thích
    3. Aggregate + rank recommendations
    4. Return top N books (không bao gồm books user đã rate)
    """
    if not MODELS_LOADED or not RATINGS_LOADED:
        return []
    
    try:
        # Bước 1: Lấy books user đã rate cao
        user_ratings = rating_df[rating_df['User-ID'] == user_id]
        
        if len(user_ratings) == 0:
            logger.warning(f"User {user_id} not found in rating data")
            return []
        
        # Filter: rating >= rating_threshold (user yêu thích)
        favorite_books = user_ratings[user_ratings['Rating'] >= rating_threshold]
        rated_isbns = set(user_ratings['ISBN'].unique())
        
        if len(favorite_books) == 0:
            logger.info(f"User {user_id}: No high-rated books (>= {rating_threshold})")
            return []
        
        logger.info(f"User {user_id}: Found {len(favorite_books)} favorite books")
        
        # Bước 2-3: Tìm similar books cho mỗi favorite
        recommendation_scores = {}
        
        for fav_isbn in favorite_books['ISBN'].values:
            if fav_isbn not in isbn_to_index:
                continue
            
            book_index = isbn_to_index[fav_isbn]
            book_vector = tfidf_matrix[book_index]
            similarities = cosine_similarity(book_vector, tfidf_matrix)[0]
            
            # Lấy top 100 similar books (để có nhiều choices)
            top_indices = np.argsort(similarities)[::-1][:100]
            
            for rank, idx in enumerate(top_indices):
                rec_isbn = index_to_isbn[idx]
                
                # Skip: chính nó, books user đã rate
                if rec_isbn == fav_isbn or rec_isbn in rated_isbns:
                    continue
                
                # Weight by similarity + inverse rank
                score = (1.0 - rank/20) * similarities[idx]
                
                if rec_isbn not in recommendation_scores:
                    recommendation_scores[rec_isbn] = 0
                recommendation_scores[rec_isbn] += score
        
        # Bước 4: Sort + return top N
        if not recommendation_scores:
            logger.warning(f"User {user_id}: No similar books found")
            return []
        
        sorted_recs = sorted(
            recommendation_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        recommended_isbns = [isbn for isbn, score in sorted_recs[:top_n]]
        
        logger.info(f"User {user_id}: Generated {len(recommended_isbns)} recommendations")
        return recommended_isbns
        
    except Exception as e:
        logger.error(f"Error generating user recommendations for {user_id}: {str(e)}")
        return []

# ============================================================
# Database Helper Functions
# ============================================================

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

# ============================================================
# API 1: Top 5 Books
# ============================================================

@app.route('/api/top5', methods=['GET'])
def get_top5():
    """
    Hot line 1: Top 5
    Returns the 5 most rated books with their details
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get top 5 ISBNs in order
        cursor.execute('SELECT isbn FROM top5')
        top5_isbns = [row['isbn'] for row in cursor.fetchall()]
        
        books = []
        for isbn in top5_isbns:
            cursor.execute('''
                SELECT isbn, title, author, year, publisher, description, image_url
                FROM books
                WHERE isbn = ?
            ''', (isbn,))
            book = cursor.fetchone()
            if book:
                books.append(book)
        
        conn.close()
        
        result = {
            'success': True,
            'count': len(books),
            'books': [dict(book) for book in books]
        }
        
        logger.info(f"Top5 API: Returned {len(books)} books")
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in top5 API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
# API 2: Search with Full-Text Search
# ============================================================

@app.route('/api/search', methods=['GET'])
def search_books():
    """
    Hot line 2: Search
    Searches for books by title or author using full-text search
    Query parameter: q (search query)
    """
    try:
        query = request.args.get('q', '').strip()
        
        if not query:
            return jsonify({'success': False, 'error': 'Query parameter "q" is required'}), 400
        
        if len(query) < 2:
            return jsonify({'success': False, 'error': 'Query must be at least 2 characters'}), 400
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Search using FTS
        cursor.execute('''
            SELECT DISTINCT books.isbn, books.title, books.author, books.year, books.image_url
            FROM books_fts
            JOIN books ON books_fts.rowid = books.rowid
            WHERE books_fts MATCH ?
            LIMIT 20
        ''', (query + '*',))  # Prefix search
        
        results = cursor.fetchall()
        conn.close()
        
        response = {
            'success': True,
            'query': query,
            'count': len(results),
            'results': [dict(book) for book in results]
        }
        
        logger.info(f"Search API: Query '{query}' returned {len(results)} results")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in search API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
# API 3: Book Details with Real-time Recommendations (TF-IDF)
# ============================================================

@app.route('/api/book/<isbn>', methods=['GET'])
def get_book_details(isbn):
    """
    Hot line 3: Book Details & Recommendations (REAL-TIME computed)
    
    Quy trình 3 bước:
    1. Query database để lấy detailsof sách
    2. Tính các sách tương tự dùng TF-IDF cosine similarity
    3. Query database để lấy metadata (title, image) của recommended books
    
    Query parameter: ?type=content (chỉ để tương thích API cũ)
    """
    try:
        recommendation_type = request.args.get('type', 'content').lower()
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # BƯỚC 1: Lấy chi tiết sách từ database
        cursor.execute('''
            SELECT isbn, title, author, year, publisher, description, image_url
            FROM books
            WHERE isbn = ?
        ''', (isbn,))
        
        book = cursor.fetchone()
        
        if not book:
            conn.close()
            return jsonify({'success': False, 'error': 'Book not found'}), 404
        
        book_dict = dict(book)
        recommendations = []
        
        # BƯỚC 2 & 3: Tính TF-IDF recommendations
        if MODELS_LOADED:
            # Tính recommendations thẳng từ model
            recommended_isbns = get_tf_idf_recommendations(isbn, top_n=5)
            
            # Lấy metadata từ database
            for rec_isbn in recommended_isbns:
                cursor.execute('''
                    SELECT isbn, title, author, year, publisher, image_url
                    FROM books
                    WHERE isbn = ?
                ''', (rec_isbn,))
                
                rec_book = cursor.fetchone()
                if rec_book:
                    recommendations.append(dict(rec_book))
            
            logger.info(f"TF-IDF recommendations: {len(recommendations)} books for ISBN {isbn}")
        else:
            logger.warning(f"Models not loaded - no recommendations available")
        
        conn.close()
        
        response = {
            'success': True,
            'book': book_dict,
            'recommendation_type': 'content-tfidf',
            'recommendations': recommendations,
            'note': 'Recommendations computed real-time using TF-IDF'
        }
        
        logger.info(f"Book details API: ISBN {isbn} with {len(recommendations)} recommendations")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in book details API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
# API 4: User-Based Recommendations
# ============================================================

@app.route('/api/user/<int:user_id>/recommendations', methods=['GET'])
def get_user_recs(user_id):
    """
    Hot line 4: User-Based Recommendations (TRẠM 3 + NGƯỜI DÙNG)
    
    Nhận user_id từ rating.csv
    Trả về TẤT CẢ sách được recommend dựa trên rating history của user
    
    Quy trình:
    1. Kiểm tra user có tồn tại không
    2. Tìm sách user yêu thích (rating >= 4)
    3. Dùng TF-IDF tìm sách tương tự
    4. Trả về TẤT CẢ recommendations với metadata từ database
    """
    try:
        if not MODELS_LOADED or not RATINGS_LOADED:
            return jsonify({
                'success': False,
                'error': 'Models or rating data not loaded'
            }), 500
        
        # Tính user recommendations - giới hạn 10 recommendations
        recommended_isbns = get_user_recommendations(user_id, top_n=10, rating_threshold=3)
        
        if not recommended_isbns:
            return jsonify({
                'success': True,
                'user_id': user_id,
                'note': 'No recommendations available for this user',
                'recommendations': []
            }), 200
        
        # Lấy metadata từ database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        recommendations = []
        for rec_isbn in recommended_isbns:
            cursor.execute('''
                SELECT isbn, title, author, year, publisher, image_url
                FROM books
                WHERE isbn = ?
            ''', (rec_isbn,))
            
            rec_book = cursor.fetchone()
            if rec_book:
                recommendations.append(dict(rec_book))
        
        # Lấy số lượng ratings của user
        user_ratings = rating_df[rating_df['User-ID'] == user_id]
        
        conn.close()
        
        response = {
            'success': True,
            'user_id': user_id,
            'user_rating_count': len(user_ratings),
            'recommendation_type': 'user-based-tfidf',
            'recommendations': recommendations,
            'note': 'Recommendations based on user rating history using TF-IDF'
        }
        
        logger.info(f"User recommendations: User {user_id} with {len(user_ratings)} ratings -> {len(recommendations)} recommendations")
        return jsonify(response), 200
        
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid user_id format (must be integer)'}), 400
    except Exception as e:
        logger.error(f"Error in user recommendations API: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================================
# Health Check
# ============================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) as count FROM books')
        count = cursor.fetchone()['count']
        conn.close()
        
        status = {
            'status': 'healthy',
            'books_in_database': count,
            'models_loaded': MODELS_LOADED,
            'ratings_loaded': RATINGS_LOADED
        }
        return jsonify(status), 200
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ============================================================
# CORS and Static Files
# ============================================================

@app.route('/', methods=['GET'])
def index():
    """Serve the main frontend page"""
    try:
        return send_file('index.html', mimetype='text/html')
    except Exception as e:
        return f'<h1>Error</h1><p>Could not load index.html: {str(e)}</p>', 500

@app.after_request
def add_cors_headers(response):
    """Add CORS headers to allow requests from frontend"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# ============================================================
# Main
# ============================================================

def signal_handler(sig, frame):
    print("\n" + "=" * 60)
    print("Shutting down Flask server...")
    print("=" * 60)
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    
    print("=" * 60)
    print("Flask Backend API Server")
    print("=" * 60)
    print("\nStarting on http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  GET /api/health                          - Health check")
    print("  GET /api/top5                            - Get top 5 books")
    print("  GE T /api/search?q=<query>                - Search for books")
    print("  GET /api/book/<isbn>                     - Get book details & recommendations")
    print("  GET /api/user/<user_id>/recommendations  - Get user-based recommendations")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
