# Collaborative Filtering Comparison Results

## Summary

This analysis compares **User-User Collaborative Filtering** vs **Item-Item Collaborative Filtering** on the movie ratings dataset.

## Methodology

- **Dataset**: ratings.csv (100,836 ratings)
- **Train/Test Split**: 70% training, 30% testing (random_state=1)
- **Similarity Metric**: Cosine Similarity
- **Evaluation Metric**: RMSE (Root Mean Squared Error)
- **Normalization**: MinMaxScaler (0.5 to 5.0 rating scale)

## Results

| Method | RMSE |
|--------|------|
| **User-User Collaborative Filtering** | **2.636169** |
| Item-Item Collaborative Filtering | 2.847238 |

## Conclusion

**User-based collaborative filtering outperforms item-based collaborative filtering.**

- Lower RMSE indicates better performance
- Difference: 0.211069 (approximately 8% improvement)

## Implementation Details

### User-User Collaborative Filtering
1. Created user-item matrix (users × movies)
2. Computed cosine similarity between users
3. Predicted ratings using: `user_similarity @ user_data`
4. Normalized predictions to [0.5, 5.0] scale
5. Calculated RMSE on test set

### Item-Item Collaborative Filtering
1. Created item-user matrix (movies × users) - transposed from user-item
2. Computed cosine similarity between items (movies)
3. Predicted ratings using: `item_similarity @ item_data`
4. Normalized predictions to [0.5, 5.0] scale
5. Transposed back to user-item format and calculated RMSE on test set

## Answer

**User-based collaborative filtering outperforms item-based collaborative filtering.**
