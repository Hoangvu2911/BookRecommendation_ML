#!/usr/bin/env python3
"""
Database utility functions for Book Recommendation System
"""

import sqlite3
from config import DATABASE_PATH
from utils.logger import setup_logger

logger = setup_logger(__name__)

def get_db_connection():
    """
    Get a database connection with row factory
    
    Returns:
        sqlite3.Connection object
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

def check_database_exists():
    """
    Check if database exists and has required tables
    
    Returns:
        bool: True if database exists and is valid
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if books table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='books'"
        )
        exists = cursor.fetchone() is not None
        
        conn.close()
        return exists
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False

def get_book_by_isbn(isbn):
    """
    Get book details by ISBN
    
    Args:
        isbn: Book ISBN
        
    Returns:
        dict: Book data or None if not found
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT isbn, title, author, year, publisher, description, image_url
            FROM books
            WHERE isbn = ?
        ''', (isbn,))
        
        book = cursor.fetchone()
        conn.close()
        
        return dict(book) if book else None
    except Exception as e:
        logger.error(f"Error fetching book {isbn}: {e}")
        return None

def search_books(query, limit=20):
    """
    Search books using full-text search
    
    Args:
        query: Search query string
        limit: Maximum number of results
        
    Returns:
        list: List of matching books
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT DISTINCT books.isbn, books.title, books.author, books.year, books.image_url
            FROM books_fts
            JOIN books ON books_fts.rowid = books.rowid
            WHERE books_fts MATCH ?
            LIMIT ?
        ''', (query + '*', limit))
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
    except Exception as e:
        logger.error(f"Error searching books with query '{query}': {e}")
        return []

def get_top5_books():
    """
    Get top 5 rated books
    
    Returns:
        list: List of top 5 books
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
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
                books.append(dict(book))
        
        conn.close()
        return books
    except Exception as e:
        logger.error(f"Error fetching top5 books: {e}")
        return []

def enrich_recommendations_with_metadata(recommendation_list):
    """
    Enrich recommendation ISBNs with book metadata from database
    
    Args:
        recommendation_list: List of dicts with 'isbn' key
        
    Returns:
        list: Enriched recommendation list with image_url, publisher, year
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        enriched = []
        for rec in recommendation_list:
            cursor.execute('''
                SELECT isbn, title, author, year, publisher, image_url
                FROM books
                WHERE isbn = ?
            ''', (rec['isbn'],))
            
            db_book = cursor.fetchone()
            if db_book:
                rec['image_url'] = db_book['image_url']
                rec['publisher'] = db_book['publisher']
                rec['year'] = db_book['year']
            
            enriched.append(rec)
        
        conn.close()
        return enriched
    except Exception as e:
        logger.error(f"Error enriching recommendations: {e}")
        return recommendation_list
