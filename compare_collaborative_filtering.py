import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the data
print("Loading data...")
ratings = pd.read_csv("ratings.csv")
print(f"Data shape: {ratings.shape}")

# Split the data: 70% training, 30% testing with random_state=1
print("\nSplitting data...")
X_train, X_test = train_test_split(ratings, test_size=0.30, random_state=1)
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# ============================================================================
# USER-USER COLLABORATIVE FILTERING
# ============================================================================
print("\n" + "="*60)
print("USER-USER COLLABORATIVE FILTERING")
print("="*60)

# Pivot ratings into user-item matrix (users as rows, movies as columns)
print("\nCreating user-item matrix...")
user_data = X_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print(f"User-item matrix shape: {user_data.shape}")

# Create dummy matrices for prediction and evaluation
dummy_train = X_train.copy()
dummy_test = X_test.copy()
dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)

dummy_train = dummy_train.pivot(index='userId', columns='movieId', values='rating').fillna(1)
dummy_test = dummy_test.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Compute user-user similarity using cosine similarity
print("Computing user-user similarity matrix...")
user_similarity = cosine_similarity(user_data)
user_similarity[np.isnan(user_similarity)] = 0
print(f"User similarity matrix shape: {user_similarity.shape}")

# Predict user ratings
print("Predicting user ratings...")
user_predicted_ratings = np.dot(user_similarity, user_data)

# Filter predictions to only movies not rated by users (for training)
user_final_ratings = np.multiply(user_predicted_ratings, dummy_train)

# Prepare test data
print("Preparing test data for user-user CF...")
test_user_features = X_test.pivot(index='userId', columns='movieId', values='rating').fillna(0)
test_user_similarity = cosine_similarity(test_user_features)
test_user_similarity[np.isnan(test_user_similarity)] = 0

# Predict on test data
user_predicted_ratings_test = np.dot(test_user_similarity, test_user_features)
test_user_final_rating = np.multiply(user_predicted_ratings_test, dummy_test)

# Normalize predictions to rating scale (0.5 to 5.0)
print("Normalizing predictions...")
X_user = test_user_final_rating.copy()
# Get all non-zero values for normalization
non_zero_values_user = X_user.values[X_user.values > 0]
if len(non_zero_values_user) > 0:
    scaler_user = MinMaxScaler(feature_range=(0.5, 5))
    scaler_user.fit(non_zero_values_user.reshape(-1, 1))
    # Transform the entire matrix, but only keep transformed values where original > 0
    X_user_values = X_user.values.copy()
    non_zero_mask = X_user_values > 0
    X_user_values[non_zero_mask] = scaler_user.transform(X_user_values[non_zero_mask].reshape(-1, 1)).flatten()
    X_user = pd.DataFrame(X_user_values, index=X_user.index, columns=X_user.columns)

# Calculate RMSE for user-user CF
test_user = X_test.pivot(index='userId', columns='movieId', values='rating')
pred_user = X_user

# Align indices and columns
common_users = test_user.index.intersection(pred_user.index)
common_movies = test_user.columns.intersection(pred_user.columns)
test_user_aligned = test_user.loc[common_users, common_movies]
pred_user_aligned = pred_user.loc[common_users, common_movies]

# Calculate RMSE only on non-NaN values
diff_sqr_matrix_user = (test_user_aligned - pred_user_aligned) ** 2
total_non_nan_user = np.count_nonzero(~np.isnan(diff_sqr_matrix_user))
sum_of_squares_err_user = np.nansum(diff_sqr_matrix_user.values)

if total_non_nan_user > 0:
    rmse_user = np.sqrt(sum_of_squares_err_user / total_non_nan_user)
else:
    rmse_user = np.inf

print(f"\nUser-User Collaborative Filtering RMSE: {rmse_user:.6f}")

# ============================================================================
# ITEM-ITEM COLLABORATIVE FILTERING
# ============================================================================
print("\n" + "="*60)
print("ITEM-ITEM COLLABORATIVE FILTERING")
print("="*60)

# Pivot ratings into item-user matrix (movies as rows, users as columns)
print("\nCreating item-user matrix...")
item_data = X_train.pivot(index='movieId', columns='userId', values='rating').fillna(0)
print(f"Item-user matrix shape: {item_data.shape}")

# Create dummy matrices for prediction and evaluation (transposed)
dummy_train_item = X_train.copy()
dummy_test_item = X_test.copy()
dummy_train_item['rating'] = dummy_train_item['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_test_item['rating'] = dummy_test_item['rating'].apply(lambda x: 1 if x > 0 else 0)

dummy_train_item = dummy_train_item.pivot(index='movieId', columns='userId', values='rating').fillna(1)
dummy_test_item = dummy_test_item.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Compute item-item similarity using cosine similarity
print("Computing item-item similarity matrix...")
item_similarity = cosine_similarity(item_data)
item_similarity[np.isnan(item_similarity)] = 0
print(f"Item similarity matrix shape: {item_similarity.shape}")

# Predict item ratings (for each user, predict based on similar items)
# For item-based CF: prediction = item_similarity * item_data
print("Predicting item ratings...")
item_predicted_ratings = np.dot(item_similarity, item_data)

# Filter predictions to only users who haven't rated items (for training)
item_final_ratings = np.multiply(item_predicted_ratings, dummy_train_item)

# Prepare test data
print("Preparing test data for item-item CF...")
test_item_features = X_test.pivot(index='movieId', columns='userId', values='rating').fillna(0)
test_item_similarity = cosine_similarity(test_item_features)
test_item_similarity[np.isnan(test_item_similarity)] = 0

# Predict on test data
item_predicted_ratings_test = np.dot(test_item_similarity, test_item_features)
test_item_final_rating = np.multiply(item_predicted_ratings_test, dummy_test_item)

# Normalize predictions to rating scale (0.5 to 5.0)
print("Normalizing predictions...")
X_item = test_item_final_rating.copy()
# Get all non-zero values for normalization
non_zero_values_item = X_item.values[X_item.values > 0]
if len(non_zero_values_item) > 0:
    scaler_item = MinMaxScaler(feature_range=(0.5, 5))
    scaler_item.fit(non_zero_values_item.reshape(-1, 1))
    # Transform the entire matrix, but only keep transformed values where original > 0
    X_item_values = X_item.values.copy()
    non_zero_mask_item = X_item_values > 0
    X_item_values[non_zero_mask_item] = scaler_item.transform(X_item_values[non_zero_mask_item].reshape(-1, 1)).flatten()
    X_item = pd.DataFrame(X_item_values, index=X_item.index, columns=X_item.columns)

# Calculate RMSE for item-item CF
# Need to transpose back to user-item format for comparison
test_item = X_test.pivot(index='movieId', columns='userId', values='rating')
pred_item = X_item

# Align indices and columns
common_movies_item = test_item.index.intersection(pred_item.index)
common_users_item = test_item.columns.intersection(pred_item.columns)
test_item_aligned = test_item.loc[common_movies_item, common_users_item]
pred_item_aligned = pred_item.loc[common_movies_item, common_users_item]

# Transpose to user-item format for RMSE calculation
test_item_aligned = test_item_aligned.T
pred_item_aligned = pred_item_aligned.T

# Calculate RMSE only on non-NaN values
diff_sqr_matrix_item = (test_item_aligned - pred_item_aligned) ** 2
total_non_nan_item = np.count_nonzero(~np.isnan(diff_sqr_matrix_item))
sum_of_squares_err_item = np.nansum(diff_sqr_matrix_item.values)

if total_non_nan_item > 0:
    rmse_item = np.sqrt(sum_of_squares_err_item / total_non_nan_item)
else:
    rmse_item = np.inf

print(f"\nItem-Item Collaborative Filtering RMSE: {rmse_item:.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"User-User Collaborative Filtering RMSE: {rmse_user:.6f}")
print(f"Item-Item Collaborative Filtering RMSE: {rmse_item:.6f}")

if rmse_user < rmse_item:
    print("\n✓ User-based collaborative filtering outperforms item-based collaborative filtering.")
    print(f"  (Lower RMSE is better. Difference: {rmse_item - rmse_user:.6f})")
elif rmse_item < rmse_user:
    print("\n✓ Item-based collaborative filtering outperforms user-based collaborative filtering.")
    print(f"  (Lower RMSE is better. Difference: {rmse_user - rmse_item:.6f})")
else:
    print("\nBoth methods have the same RMSE.")
