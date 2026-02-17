#!/usr/bin/env python3
"""
TRẠM 1: Export Models to .pkl files
Từ Model.ipynb, loại bỏ xuất CSV, chỉ giữ:
- TF-IDF Ma trận & Vectorizer
- Bản đồ ISBN (Index Map)
- SVD Model (Collaborative Filtering)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import cross_validate
import os

print("=" * 70)
print("TRẠM 1: EXPORT MODELS TO .PKL (Đóng gói Não bộ)")
print("=" * 70)

# ============================================================================
# BƯỚC 1: LOAD DỮ LIỆU
# ============================================================================
print("\n1. Loading dữ liệu...")
# Load book data từ data/des_cross.csv hoặc data/book.csv
try:
    book_df = pd.read_csv('data/des_cross.csv', usecols=['ISBN', 'Title', 'Author', 'description'])
    # Load genres từ data/book.csv
    book_genres = pd.read_csv('data/book.csv', usecols=['ISBN', 'genres'])
    book_df = pd.merge(book_df, book_genres, on='ISBN', how='left')
except:
    book_df = pd.read_csv('data/book.csv', usecols=['ISBN', 'Title', 'Author', 'description', 'genres'])

rating_df = pd.read_csv('data/rating.csv')
print(f"   ✓ Books: {len(book_df)}")
print(f"   ✓ Ratings: {len(rating_df)}")

# ============================================================================
# BƯỚC 2: TF-IDF (Content-Based)
# ============================================================================
print("\n2. Training TF-IDF...")

# Xóa duplicates
book_df = book_df.drop_duplicates(subset=['ISBN'], keep='first')
book_df['description'] = book_df['description'].fillna('')
book_df['genres'] = book_df['genres'].fillna('')

# Hàm transform Author
def author_transform(x):
    if isinstance(x, str):
        return x.replace(" ", "").replace(".", "").replace(",", "").replace("-", "").lower()
    return ""

# Hàm transform genres (giống Model.ipynb)
def transform_genre(x):
    if isinstance(x, str):
        elements = x.split(",")
        cleaned_elements = []
        for e in elements:
            clean_text = e.lower().replace(" ", "").replace("-", "").replace("'", "")
            cleaned_elements.append(clean_text)
        return " ".join(cleaned_elements)
    else:
        return ""

# Tạo feature text (Title + Author + genres + Description) - giống Model.ipynb
book_df['feature'] = (
    book_df['Title'].astype(str).str.lower() + " " +
    book_df['Author'].apply(author_transform) + " " +
    book_df['genres'].apply(transform_genre) + " " +
    book_df['description'].astype(str).str.lower()
)

# Train TF-IDF vectorizer (giống Model.ipynb)
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=2
)
tfidf_matrix = tfidf_vectorizer.fit_transform(book_df['feature'])
print(f"   ✓ TF-IDF Matrix shape: {tfidf_matrix.shape}")
print(f"   ✓ TF-IDF features: {tfidf_matrix.shape[1]}")

# ============================================================================
# BƯỚC 3: Bản đồ ISBN (Index Map)
# ============================================================================
print("\n3. Creating ISBN Index Map...")
isbn_list = book_df['ISBN'].values
isbn_to_index = {isbn: idx for idx, isbn in enumerate(isbn_list)}
index_to_isbn = {idx: isbn for isbn, idx in isbn_to_index.items()}
print(f"   ✓ ISBN Map created: {len(isbn_to_index)} books")

# ============================================================================
# BƯỚC 4: KNN (Collaborative Filtering - Item-Item)
# ============================================================================
print("\n4. Training KNN Model...")

# Create dataset for surprise
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(rating_df[['User-ID', 'ISBN', 'Rating']], reader)

# Tìm K tốt nhất bằng cross_validate (giống Model.ipynb)
print("   Searching for best K...")
sim_options = {
    'name': 'pearson',
    'user_based': False
}
k_candidates = list(range(10, 50, 1))
rmse_knn = []

for k in k_candidates:
    algo = KNNWithMeans(k=k, min_k=1, sim_options=sim_options)
    result = cross_validate(algo, data, measures=['RMSE'], cv=5, verbose=False)
    mean_rmse = result['test_rmse'].mean()
    rmse_knn.append(mean_rmse)

best_k = k_candidates[rmse_knn.index(min(rmse_knn))]
print(f"   ✓ Best K found: {best_k} (RMSE: {min(rmse_knn):.4f})")

# Train KNN với K tốt nhất
full_train = data.build_full_trainset()
knn_model = KNNWithMeans(k=best_k, min_k=1, sim_options=sim_options)
knn_model.fit(full_train)
print(f"   ✓ KNN Model trained (k={best_k})")

# ============================================================================
# BƯỚC 5: EXPORT thành .pkl FILES
# ============================================================================
print("\n5. Exporting to .pkl files...")

# Export TF-IDF Vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print(f"   ✓ Saved: tfidf_vectorizer.pkl")

# Export TF-IDF Matrix
joblib.dump(tfidf_matrix, 'tfidf_matrix.pkl')
print(f"   ✓ Saved: tfidf_matrix.pkl ({tfidf_matrix.data.nbytes / 1024 / 1024:.2f} MB)")

# Export ISBN Maps
isbn_map = {'isbn_to_index': isbn_to_index, 'index_to_isbn': index_to_isbn}
joblib.dump(isbn_map, 'isbn_map.pkl')
print(f"   ✓ Saved: isbn_map.pkl")

# Export KNN Model
joblib.dump(knn_model, 'knn_model.pkl')
print(f"   ✓ Saved: knn_model.pkl")

print("\n" + "=" * 70)
print("✓ TRẠM 1 HOÀN TẤT!")
print("=" * 70)
print("\nCác file đã tạo:")
print("  1. tfidf_vectorizer.pkl  - Vectorizer để transform text (với genres)")
print("  2. tfidf_matrix.pkl      - Ma trận TF-IDF toàn bộ sách")
print("  3. isbn_map.pkl          - Bản đồ ISBN ↔ Index")
print("  4. knn_model.pkl         - Mô hình KNN (Item-Item collaborative filtering)")
print("\nNhớ di chuyển 4 file này vào thư mục project trước khi chạy app.py!")
