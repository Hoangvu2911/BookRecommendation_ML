"""
Create SQLite Database with Full-Text Search
"""

import sqlite3
import pandas as pd
import os

print("=" * 60)
print("Creating SQLite Database with Full-Text Search")
print("=" * 60)

# Remove existing database if it exists
if os.path.exists('books_data.db'):
    os.remove('books_data.db')
    print("\nRemoved existing database")

# Connect to database
conn = sqlite3.connect('books_data.db')
cursor = conn.cursor()

# Load data
print("\n1. Loading data from CSV files...")

# Load top 5
top5_file = 'data/top5.csv'
if os.path.exists(top5_file):
    top5_df = pd.read_csv(top5_file, header=None, names=['ISBN'])
else:
    print(f"   {top5_file} not found, computing top 5 from ratings...")
    # Compute top 5 books by average rating
    rating_df = pd.read_csv('data/rating.csv')
    top5_books = rating_df.groupby('ISBN').agg({
        'Rating': ['mean', 'count']
    }).reset_index()
    top5_books.columns = ['ISBN', 'avg_rating', 'count']
    top5_books = top5_books[top5_books['count'] >= 5]  # At least 5 ratings
    top5_books = top5_books.nlargest(5, 'avg_rating')
    top5_df = top5_books[['ISBN']].copy()
    print(f"   Computed top 5 books from {len(rating_df)} ratings")

des_cross_df = pd.read_csv('data/des_cross.csv')
des_book_df = pd.read_csv('data/des_book.csv')  # Contains Image URLs

print(f"   Top 5: {len(top5_df)} books")
print(f"   Books data: {len(des_cross_df)} books")
print(f"   Book images (des_book): {len(des_book_df)} books")
print("   Note: Recommendations will be computed real-time using .pkl models")

# Merge image URLs from des_book
print("   Merging image URLs...")
image_url_map = {}
for idx, row in des_book_df.iterrows():
    image_url_map[row['ISBN']] = row.get('Image-URL-M', None) or row.get('Image-URL-L', None) or row.get('Image-URL-S', None)
print(f"   Found {len(image_url_map)} image URLs")

# Create books table with full description
print("\n2. Creating books table...")
cursor.execute('''
CREATE TABLE IF NOT EXISTS books (
    isbn TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT,
    year INTEGER,
    publisher TEXT,
    description TEXT,
    image_url TEXT
)
''')

# Insert books data
print("   Inserting book data...")
for idx, row in des_cross_df.iterrows():
    try:
        isbn = row['ISBN']
        image_url = image_url_map.get(isbn, None)
        cursor.execute('''
            INSERT OR REPLACE INTO books (isbn, title, author, year, publisher, description, image_url)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            isbn,
            row['Title'],
            row['Author'],
            int(row['Year']) if pd.notna(row['Year']) and row['Year'] != 0 else None,
            row['Publisher'] if 'Publisher' in row.index else None,
            row['description'],
            image_url
        ))
        if (idx + 1) % 1000 == 0:
            print(f"      Inserted {idx + 1}/{len(des_cross_df)} books...")
    except Exception as e:
        pass

conn.commit()
print(f"   Inserted {len(des_cross_df)} books")

# Create FTS (Full-Text Search) virtual table
print("\n3. Creating Full-Text Search index...")
cursor.execute('''
CREATE VIRTUAL TABLE IF NOT EXISTS books_fts USING fts5(
    isbn,
    title,
    author,
    content=books,
    content_rowid=rowid
)
''')

# Populate FTS table
cursor.execute('''
INSERT INTO books_fts (rowid, isbn, title, author)
SELECT rowid, isbn, title, author FROM books
''')
conn.commit()
print("   FTS index created")

# Create top5 table
print("\n4. Creating top5 table...")
cursor.execute('''
CREATE TABLE IF NOT EXISTS top5 (
    isbn TEXT PRIMARY KEY
)
''')

for idx, row in top5_df.iterrows():
    cursor.execute('INSERT INTO top5 (isbn) VALUES (?)', (row['ISBN'],))

conn.commit()
print(f"   Inserted {len(top5_df)} top books")

# Create indexes for faster queries
print("\n5. Creating indexes...")
cursor.execute('CREATE INDEX IF NOT EXISTS idx_books_isbn ON books(isbn)')
conn.commit()
print("   Indexes created")

# Test the database
print("\n6. Testing database queries...")

# Test 1: Get top 5 books
cursor.execute('''
SELECT b.isbn, b.title, b.author FROM books b
WHERE b.isbn IN (SELECT isbn FROM top5)
''')
top5_books = cursor.fetchall()
print(f"   Top 5 test: Found {len(top5_books)} books")
for isbn, title, author in top5_books:
    print(f"      {isbn}: {title} by {author}")

# Test 2: Full-text search
cursor.execute('''
SELECT books.isbn, books.title, books.author FROM books_fts
JOIN books ON books_fts.rowid = books.rowid
WHERE books_fts MATCH 'Harry'
LIMIT 3
''')
search_results = cursor.fetchall()
print(f"   Search test: Found {len(search_results)} results for 'Harry'")

conn.close()

print("\n" + "=" * 60)
print("DATABASE CREATED SUCCESSFULLY")
print("=" * 60)
print("\nDatabase file: books_data.db")
print("Tables created:")
print("  books: Main book information with FTS search (metadata only)")
print("  books_fts: Full-text search index")
print("  top5: Top 5 rated books")
print("\nTables NOT included:")
print("  recommendations: Computed real-time using models")
print("\nModel files required at runtime (3 files from export_models.py):")
print("  1. cosine_sim.pkl           - Cosine similarity matrix")
print("  2. knn_model.pkl            - KNN model (Item-Item CF)")
print("  3. svd_model.pkl            - SVD model (Matrix Factorization)")
print("\nRecommendation Strategy:")
print("  final_score = 0.2 * content_based + 0.3 * knn + 0.5 * svd")
print("\nReady for server startup with combined recommendation engine!")
