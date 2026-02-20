#!/usr/bin/env python3
"""
Recommender Class - Combines Content-Based, KNN, and SVD Models
Implements the combined scoring mechanism from Model.ipynb
"""

import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class Recommender:
    """
    Combined Recommendation Engine using:
    1. Content-Based (TF-IDF + Cosine Similarity) - Weight: 0.2
    2. KNN (Item-Item Collaborative Filtering) - Weight: 0.3
    3. SVD (Matrix Factorization) - Weight: 0.5
    """
    
    def __init__(self, book_df, rating_df):
        """
        Initialize the recommender with data and load models
        
        Args:
            book_df: DataFrame with columns [ISBN, Title, Author, description, genres]
            rating_df: DataFrame with columns [User-ID, ISBN, Rating]
        """
        self.book_df = book_df
        self.rating_df = rating_df
        
        # Load pre-trained models from pkl folder
        print("Loading pre-trained models from pkl folder...")
        try:
            # Path to pkl folder
            pkl_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pkl')
            
            self.cosine_sim = joblib.load(os.path.join(pkl_folder, 'cosine_sim.pkl'))
            self.knn_model = joblib.load(os.path.join(pkl_folder, 'knn_model.pkl'))
            self.svd_model = joblib.load(os.path.join(pkl_folder, 'svd_model.pkl'))
            print("All models loaded successfully")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print(f"Please download .pkl files from Google Drive and place them in: {pkl_folder}/")
            raise
    
    def get_user_history(self, user_id):
        """
        Get the user's rating history
        
        Args:
            user_id: User ID
            
        Returns:
            DataFrame with user's ratings and corresponding book indices
        """
        user_rating_list = self.rating_df[self.rating_df['User-ID'] == user_id][['ISBN', 'Rating']]
        
        if len(user_rating_list) == 0:
            print(f"User {user_id} has no ratings in the system")
            return pd.DataFrame()
        
        # Map ISBN to book index
        book_indices = pd.DataFrame({
            'ISBN': self.book_df['ISBN'].values,
            'matrix_index': range(len(self.book_df))
        })
        
        user_df = pd.merge(user_rating_list, book_indices, on='ISBN', how='inner')
        return user_df
    
    def calculate_content_score(self, book_idx, user_df):
        """
        Calculate content-based score for a book based on user's history
        
        Args:
            book_idx: Index of the book in the matrix
            user_df: User's rating history with book indices
            
        Returns:
            Normalized content-based score
        """
        if len(user_df) == 0:
            return 0
        
        score = 0
        sum_sim = 0
        
        # Weighted sum of similarities with rated books
        for j in range(len(user_df)):
            if book_idx != user_df['matrix_index'].iloc[j]:
                similarity = self.cosine_sim[book_idx][user_df['matrix_index'].iloc[j]]
                rating = user_df['Rating'].iloc[j]
                score += rating * similarity
                sum_sim += similarity
        
        # Avoid division by zero
        if sum_sim == 0:
            return 0
        
        return score / sum_sim
    
    def calculate_knn_score(self, user_id, isbn):
        """
        Get KNN predicted rating for user-book pair
        
        Args:
            user_id: User ID
            isbn: Book ISBN
            
        Returns:
            KNN predicted rating score
        """
        try:
            prediction = self.knn_model.predict(user_id, isbn)
            return prediction.est
        except:
            return 0
    
    def calculate_svd_score(self, user_id, isbn):
        """
        Get SVD predicted rating for user-book pair
        
        Args:
            user_id: User ID
            isbn: Book ISBN
            
        Returns:
            SVD predicted rating score
        """
        try:
            prediction = self.svd_model.predict(user_id, isbn)
            return prediction.est
        except:
            return 0
    
    def get_recommendations(self, user_id, n=10, verbose=False):
        """
        Get top N book recommendations for a user using combined scoring
        
        Combined Score Formula (matches Model.ipynb):
        if sum_sim == 0:
            final_score = 0 + 0.3 * knn + 0.5 * svd
        else:
            final_score = 0.2 * content_based + 0.3 * knn + 0.5 * svd
        
        Args:
            user_id: User ID
            n: Number of recommendations to return
            verbose: Print detailed scoring information
            
        Returns:
            List of tuples (score, isbn, title, author) sorted by score
        """
        user_df = self.get_user_history(user_id)
        
        # Get books the user has already rated (to exclude from recommendations)
        rated_isbns = set(user_df['ISBN'].values) if len(user_df) > 0 else set()
        
        scores = []
        
        for idx in range(len(self.book_df)):
            isbn = self.book_df['ISBN'].iloc[idx]
            
            # Skip already-rated books
            if isbn in rated_isbns:
                continue
            
            # Calculate scores from three models
            content_score = self.calculate_content_score(idx, user_df)  # No scaling - already 0-10 range
            knn_score = self.calculate_knn_score(user_id, isbn)
            svd_score = self.calculate_svd_score(user_id, isbn)
            
            # Combine scores with weights
            final_score = (0.2 * content_score + 
                          0.3 * knn_score + 
                          0.5 * svd_score)
            
            title = self.book_df['Title'].iloc[idx]
            author = self.book_df['Author'].iloc[idx]
            
            if verbose and final_score > 0:
                print(f"{title} (ISBN: {isbn})")
                print(f"  Content: {content_score:.2f} | KNN: {knn_score:.2f} | SVD: {svd_score:.2f} | Final: {final_score:.2f}")
            
            scores.append((final_score, isbn, title, author))
        
        # Sort by score descending and return top N
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:n]
    
    def recommend_for_user(self, user_id, n=10):
        """
        Simple interface to get recommendations for a user
        
        Args:
            user_id: User ID
            n: Number of recommendations
            
        Returns:
            List of dicts with recommendation info
        """
        recommendations = self.get_recommendations(user_id, n)
        
        result = []
        for score, isbn, title, author in recommendations:
            result.append({
                'isbn': isbn,
                'title': title,
                'author': author,
                'score': round(score, 2),
                'rating': round(score, 1)  # Interpret score as 1-10 rating
            })
        
        return result


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING RECOMMENDER CLASS")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    book_df = pd.read_csv('data/book.csv', usecols=['ISBN', 'Title', 'Author', 'description', 'genres'])
    rating_df = pd.read_csv('data/rating.csv')
    print(f"   Loaded {len(book_df)} books and {len(rating_df)} ratings")
    
    # Initialize recommender
    print("\n2. Initializing recommender...")
    recommender = Recommender(book_df, rating_df)
    
    # Test with a user
    test_user_id = 87712
    print(f"\n3. Getting recommendations for User {test_user_id}...")
    
    recommendations = recommender.recommend_for_user(test_user_id, n=10)
    
    print(f"\n   Top 10 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['title']}")
        print(f"      ISBN: {rec['isbn']} | Score: {rec['score']:.2f}/10")
    
    print("\n" + "=" * 70)
    print("RECOMMENDER TEST COMPLETE")
    print("=" * 70)
