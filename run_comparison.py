#!/usr/bin/env python3
"""
Collaborative Filtering: User-Based vs Item-Based Comparison

This script compares user-based and item-based collaborative filtering using:
- Random state: 1
- Train/Test split: 70/30
- Evaluation metric: RMSE
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("COLLABORATIVE FILTERING COMPARISON")
print("="*70)

# Load the ratings data
print("\n1. Loading data...")
ratings = pd.read_csv('ratings.csv')
print(f"   Dataset shape: {ratings.shape}")

# Split data into train and test (70/30 split with random_state=1)
print("\n2. Splitting data (70/30, random_state=1)...")
X_train, X_test = train_test_split(ratings, test_size=0.30, random_state=1)
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# Create dummy datasets for masking
dummy_train = X_train.copy()
dummy_test = X_test.copy()

print("\n" + "="*70)
print("USER-BASED COLLABORATIVE FILTERING")
print("="*70)

# Pivot ratings: Users as rows, Movies as columns
print("\n3. Creating user-item matrix...")
user_data = X_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print(f"   User-item matrix shape: {user_data.shape}")

# Prepare dummy matrices for user-based filtering
dummy_train_user = dummy_train.pivot(index='userId', columns='movieId', values='rating').fillna(1)
dummy_test_user = dummy_test.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Calculate user-user similarity using cosine similarity
print("\n4. Computing user-user similarity...")
user_similarity = cosine_similarity(user_data)
user_similarity[np.isnan(user_similarity)] = 0
print(f"   Similarity matrix shape: {user_similarity.shape}")

# Prepare test user features
print("\n5. Preparing test features...")
test_user_features = X_test.pivot(index='userId', columns='movieId', values='rating').fillna(0)
test_user_similarity = cosine_similarity(test_user_features)
test_user_similarity[np.isnan(test_user_similarity)] = 0

# Predict ratings for test set using user-based CF
print("\n6. Making predictions...")
user_predicted_ratings_test = np.dot(test_user_similarity, test_user_features)
test_user_final_rating = np.multiply(user_predicted_ratings_test, dummy_test_user)

# Scale predictions to rating range (0.5 to 5)
print("\n7. Scaling predictions...")
X = test_user_final_rating.copy()
X = X[X > 0]  # Only consider non-zero values

scaler = MinMaxScaler(feature_range=(0.5, 5))
scaler.fit(X)
user_pred = scaler.transform(X)

# Calculate RMSE for user-based CF
print("\n8. Computing RMSE...")
test_user = X_test.pivot(index='userId', columns='movieId', values='rating')

diff_sqr_matrix = (test_user - user_pred)**2
sum_of_squares_err = diff_sqr_matrix.sum().sum()
total_non_nan = np.count_nonzero(~np.isnan(user_pred))

user_rmse = np.sqrt(sum_of_squares_err / total_non_nan)
print(f"\n   USER-BASED CF RMSE: {user_rmse:.6f}")

print("\n" + "="*70)
print("ITEM-BASED COLLABORATIVE FILTERING")
print("="*70)

# Pivot ratings: Movies as rows, Users as columns (transpose of user-based)
print("\n9. Creating item-user matrix...")
item_data = X_train.pivot(index='movieId', columns='userId', values='rating').fillna(0)
print(f"   Item-user matrix shape: {item_data.shape}")

# Prepare dummy matrices for item-based filtering
dummy_train_item = dummy_train.pivot(index='movieId', columns='userId', values='rating').fillna(1)
dummy_test_item = dummy_test.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Calculate item-item similarity using cosine similarity
print("\n10. Computing item-item similarity...")
item_similarity = cosine_similarity(item_data)
item_similarity[np.isnan(item_similarity)] = 0
print(f"    Similarity matrix shape: {item_similarity.shape}")

# Prepare test item features
print("\n11. Preparing test features...")
test_item_features = X_test.pivot(index='movieId', columns='userId', values='rating').fillna(0)
test_item_similarity = cosine_similarity(test_item_features)
test_item_similarity[np.isnan(test_item_similarity)] = 0

# Predict ratings for test set using item-based CF
print("\n12. Making predictions...")
item_predicted_ratings_test = np.dot(test_item_similarity, test_item_features)
test_item_final_rating = np.multiply(item_predicted_ratings_test, dummy_test_item)

# Scale predictions to rating range (0.5 to 5)
print("\n13. Scaling predictions...")
Y = test_item_final_rating.copy()
Y = Y[Y > 0]  # Only consider non-zero values

scaler_item = MinMaxScaler(feature_range=(0.5, 5))
scaler_item.fit(Y)
item_pred = scaler_item.transform(Y)

# Calculate RMSE for item-based CF
print("\n14. Computing RMSE...")
test_item = X_test.pivot(index='movieId', columns='userId', values='rating')

diff_sqr_matrix_item = (test_item - item_pred)**2
sum_of_squares_err_item = diff_sqr_matrix_item.sum().sum()
total_non_nan_item = np.count_nonzero(~np.isnan(item_pred))

item_rmse = np.sqrt(sum_of_squares_err_item / total_non_nan_item)
print(f"\n    ITEM-BASED CF RMSE: {item_rmse:.6f}")

# Compare the two approaches
print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)
print(f"\nUser-Based CF RMSE: {user_rmse:.6f}")
print(f"Item-Based CF RMSE: {item_rmse:.6f}")
print(f"\nDifference: {abs(user_rmse - item_rmse):.6f}")

if user_rmse < item_rmse:
    print(f"\n✓ USER-BASED collaborative filtering performs BETTER (lower RMSE)")
    print(f"  User-based is {((item_rmse - user_rmse) / item_rmse * 100):.2f}% better")
    print(f"\nANSWER: User-based collaborative filtering outperforms item-based collaborative filtering.")
else:
    print(f"\n✓ ITEM-BASED collaborative filtering performs BETTER (lower RMSE)")
    print(f"  Item-based is {((user_rmse - item_rmse) / user_rmse * 100):.2f}% better")
    print(f"\nANSWER: Item-based collaborative filtering outperforms user-based collaborative filtering.")

print("="*70)
