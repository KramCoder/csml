import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load the data
file_path = "ratings.csv"
ratings = pd.read_csv(file_path)

# Split the data: 70% training, 30% testing with random_state=1
X_train, X_test = train_test_split(ratings, test_size=0.30, random_state=1)

print("Data shapes:")
print(f"Training: {X_train.shape}")
print(f"Testing: {X_test.shape}")

# ============================================================================
# USER-USER COLLABORATIVE FILTERING
# ============================================================================
print("\n" + "="*60)
print("USER-USER COLLABORATIVE FILTERING")
print("="*60)

# Pivot ratings into user-item matrix (users as rows, movies as columns)
user_data = X_train.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Create dummy matrices for train and test
dummy_train = X_train.copy()
dummy_test = X_test.copy()
dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_test['rating'] = dummy_test['rating'].apply(lambda x: 1 if x > 0 else 0)

dummy_train = dummy_train.pivot(index='userId', columns='movieId', values='rating').fillna(1)
dummy_test = dummy_test.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Calculate user-user similarity using cosine similarity
user_similarity = cosine_similarity(user_data)
user_similarity[np.isnan(user_similarity)] = 0

# Predict user ratings on movies
user_predicted_ratings = np.dot(user_similarity, user_data)

# For test evaluation: create test user features and similarity
test_user_features = X_test.pivot(index='userId', columns='movieId', values='rating').fillna(0)
test_user_similarity = cosine_similarity(test_user_features)
test_user_similarity[np.isnan(test_user_similarity)] = 0

# Predict test ratings
user_predicted_ratings_test = np.dot(test_user_similarity, test_user_features)

# Filter to only rated movies in test set
test_user_final_rating = np.multiply(user_predicted_ratings_test, dummy_test)

# Calculate RMSE for user-user CF
test = X_test.pivot(index='userId', columns='movieId', values='rating')

# Align test_user_final_rating with test structure
test_user_final_rating_df = pd.DataFrame(
    test_user_final_rating, 
    index=test_user_features.index, 
    columns=test_user_features.columns
)
test_user_final_rating_aligned = test_user_final_rating_df.reindex(
    index=test.index, 
    columns=test.columns, 
    fill_value=0
)

# Scale predictions to rating range (0.5 to 5)
X = test_user_final_rating_aligned.copy()

# Only consider values where test is not NaN (i.e., where user actually rated the movie)
test_not_nan_mask = ~test.isna()
X_filtered = X[test_not_nan_mask & (X > 0)]  # Only consider non-zero values where test exists

scaler = MinMaxScaler(feature_range=(0.5, 5))
scaler.fit(X_filtered.values.reshape(-1, 1))

# Transform only the values where test is not NaN
pred_user_df = X.copy()
non_zero_and_rated_mask = test_not_nan_mask & (X > 0)

# Get the values to transform using numpy boolean indexing
pred_user_array = pred_user_df.values.copy()
mask_array = non_zero_and_rated_mask.values
X_to_transform = pred_user_array[mask_array].reshape(-1, 1)
pred_transformed = scaler.transform(X_to_transform)

# Assign scaled values back using the same mask
pred_user_array[mask_array] = pred_transformed.flatten()
pred_user_df = pd.DataFrame(pred_user_array, index=X.index, columns=X.columns)

# Set predictions to NaN where test has NaN (unrated movies)
pred_user_df[test.isna()] = np.nan

total_non_nan_user = np.count_nonzero(~np.isnan(pred_user_df))
diff_sqr_matrix_user = (test - pred_user_df)**2
sum_of_squares_err_user = diff_sqr_matrix_user.sum().sum()
rmse_user = np.sqrt(sum_of_squares_err_user / total_non_nan_user)

print(f"User-User CF RMSE: {rmse_user:.6f}")

# ============================================================================
# ITEM-ITEM COLLABORATIVE FILTERING
# ============================================================================
print("\n" + "="*60)
print("ITEM-ITEM COLLABORATIVE FILTERING")
print("="*60)

# Pivot ratings into item-user matrix (movies as rows, users as columns)
item_data = X_train.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Create dummy matrices for item-item CF (transposed perspective)
# For items: dummy_train marks items NOT rated by users as 1, rated as 0
# For items: dummy_test marks items rated by users as 1, not rated as 0
dummy_train_item = X_train.copy()
dummy_test_item = X_test.copy()
dummy_train_item['rating'] = dummy_train_item['rating'].apply(lambda x: 0 if x > 0 else 1)
dummy_test_item['rating'] = dummy_test_item['rating'].apply(lambda x: 1 if x > 0 else 0)

dummy_train_item = dummy_train_item.pivot(index='movieId', columns='userId', values='rating').fillna(1)
dummy_test_item = dummy_test_item.pivot(index='movieId', columns='userId', values='rating').fillna(0)

# Calculate item-item similarity using cosine similarity
item_similarity = cosine_similarity(item_data)
item_similarity[np.isnan(item_similarity)] = 0

# Predict ratings: for item-item CF, we use item_data @ item_similarity
# This gives us predicted ratings for each item by each user
item_predicted_ratings = np.dot(item_similarity, item_data)

# For test evaluation: create test item features and similarity
test_item_features = X_test.pivot(index='movieId', columns='userId', values='rating').fillna(0)
test_item_similarity = cosine_similarity(test_item_features)
test_item_similarity[np.isnan(test_item_similarity)] = 0

# Predict test ratings
item_predicted_ratings_test = np.dot(test_item_similarity, test_item_features)

# Filter to only rated items in test set
test_item_final_rating = np.multiply(item_predicted_ratings_test, dummy_test_item)

# Convert to DataFrame and transpose back to user-item format for evaluation (users as rows, movies as columns)
test_item_final_rating_df = pd.DataFrame(
    test_item_final_rating, 
    index=test_item_features.index, 
    columns=test_item_features.columns
)
test_item_final_rating_transposed = test_item_final_rating_df.T

# Align with test DataFrame (ensure same users and movies)
test_item_final_rating_aligned = test_item_final_rating_transposed.reindex(
    index=test.index, 
    columns=test.columns, 
    fill_value=0
)

# Scale predictions to rating range (0.5 to 5)
X_item = test_item_final_rating_aligned.copy()

# Only consider values where test is not NaN (i.e., where user actually rated the movie)
test_not_nan_mask_item = ~test.isna()
X_item_filtered = X_item[test_not_nan_mask_item & (X_item > 0)]  # Only consider non-zero values where test exists

if len(X_item_filtered) > 0:
    scaler_item = MinMaxScaler(feature_range=(0.5, 5))
    scaler_item.fit(X_item_filtered.values.reshape(-1, 1))
    
    # Transform only the values where test is not NaN
    pred_item_df = X_item.copy()
    non_zero_and_rated_mask_item = test_not_nan_mask_item & (X_item > 0)
    
    # Get the values to transform using numpy boolean indexing
    pred_item_array = pred_item_df.values.copy()
    mask_array_item = non_zero_and_rated_mask_item.values
    X_item_to_transform = pred_item_array[mask_array_item].reshape(-1, 1)
    pred_item_transformed = scaler_item.transform(X_item_to_transform)
    
    # Assign scaled values back using the same mask
    pred_item_array[mask_array_item] = pred_item_transformed.flatten()
    pred_item_df = pd.DataFrame(pred_item_array, index=X_item.index, columns=X_item.columns)
    
    # Set predictions to NaN where test has NaN (unrated movies)
    pred_item_df[test.isna()] = np.nan
else:
    pred_item_df = test_item_final_rating_aligned.copy()
    pred_item_df[test.isna()] = np.nan

# Calculate RMSE for item-item CF
total_non_nan_item = np.count_nonzero(~np.isnan(pred_item_df))
diff_sqr_matrix_item = (test - pred_item_df)**2
sum_of_squares_err_item = diff_sqr_matrix_item.sum().sum()
rmse_item = np.sqrt(sum_of_squares_err_item / total_non_nan_item)

print(f"Item-Item CF RMSE: {rmse_item:.6f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"User-User CF RMSE:   {rmse_user:.6f}")
print(f"Item-Item CF RMSE:   {rmse_item:.6f}")
print(f"\nDifference:          {abs(rmse_user - rmse_item):.6f}")

if rmse_user < rmse_item:
    print("\n✓ User-based collaborative filtering outperforms item-based collaborative filtering.")
    print(f"  (Lower RMSE is better: {rmse_user:.6f} < {rmse_item:.6f})")
else:
    print("\n✓ Item-based collaborative filtering outperforms user-based collaborative filtering.")
    print(f"  (Lower RMSE is better: {rmse_item:.6f} < {rmse_user:.6f})")
