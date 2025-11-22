import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Import the heart disease dataset
dataset = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','class']

# Read and clean data
dataset = pd.read_csv(dataset, names=column_names)
dataset = dataset.replace('?', np.nan)
dataset = dataset.apply(pd.to_numeric, errors='coerce')
data = dataset.dropna()

# Create X and Y datasets
X = data.iloc[:,0:13]
y = data.iloc[:,-1]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=90)

print("="*60)
print("HEART DISEASE CLASSIFICATION COMPARISON")
print("="*60)
print(f"Dataset: {data.shape[0]} samples, {X.shape[1]} features")
print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
print(f"Original class distribution: {dict(y.value_counts().sort_index())}")

# =============================================================================
# MULTI-CLASS CLASSIFICATION MODEL
# =============================================================================
print("\n" + "="*40)
print("MULTI-CLASS CLASSIFICATION")
print("="*40)

# Prepare data for multi-class classification
y_train_multiclass = to_categorical(y_train)
y_test_multiclass = to_categorical(y_test)

# Create multi-class model
model_multiclass = Sequential()
model_multiclass.add(Input(shape=(13,)))
model_multiclass.add(Dense(10, kernel_initializer='normal', activation='relu'))
model_multiclass.add(Dense(8, kernel_initializer='normal', activation='relu'))
model_multiclass.add(Dense(4, kernel_initializer='normal', activation='relu'))
model_multiclass.add(Dense(5, activation='softmax'))

# Compile and train multi-class model
model_multiclass.compile(loss='categorical_crossentropy', 
                        optimizer=Adam(learning_rate=0.001), 
                        metrics=['accuracy'])

print("Training multi-class model...")
history_multiclass = model_multiclass.fit(X_train, y_train_multiclass, 
                                         epochs=60, batch_size=8, verbose=0, 
                                         validation_data=(X_test, y_test_multiclass))

# Evaluate multi-class model
pred_multiclass = model_multiclass.predict(X_test, verbose=0)
y_pred_multiclass = np.argmax(pred_multiclass, axis=1)
y_test_multiclass_labels = np.argmax(y_test_multiclass, axis=1)

multiclass_accuracy = accuracy_score(y_test_multiclass_labels, y_pred_multiclass)
print(f"Multi-class Accuracy: {multiclass_accuracy * 100:.2f}%")

# =============================================================================
# BINARY CLASSIFICATION MODEL
# =============================================================================
print("\n" + "="*40)
print("BINARY CLASSIFICATION")
print("="*40)

# Convert to binary classification: 0 = no disease, 1 = disease (any level)
y_binary = (y > 0).astype(int)
y_train_binary = y_binary[y_train.index]
y_test_binary = y_binary[y_test.index]

print(f"Binary class distribution: No disease={np.sum(y_binary == 0)}, Disease={np.sum(y_binary == 1)}")

# Create binary model
model_binary = Sequential()
model_binary.add(Input(shape=(13,)))
model_binary.add(Dense(10, kernel_initializer='normal', activation='relu'))
model_binary.add(Dense(8, kernel_initializer='normal', activation='relu'))
model_binary.add(Dense(4, kernel_initializer='normal', activation='relu'))
model_binary.add(Dense(1, activation='sigmoid'))  # Binary classification with sigmoid

# Compile and train binary model
model_binary.compile(loss='binary_crossentropy', 
                    optimizer=Adam(learning_rate=0.001), 
                    metrics=['accuracy'])

print("Training binary model...")
history_binary = model_binary.fit(X_train, y_train_binary, 
                                 epochs=60, batch_size=8, verbose=0, 
                                 validation_data=(X_test, y_test_binary))

# Evaluate binary model
pred_binary = model_binary.predict(X_test, verbose=0)
y_pred_binary = (pred_binary > 0.5).astype(int).flatten()

binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"Binary Accuracy: {binary_accuracy * 100:.2f}%")

# =============================================================================
# COMPARISON AND ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)

print(f"Multi-class Classification Accuracy: {multiclass_accuracy * 100:.2f}%")
print(f"Binary Classification Accuracy: {binary_accuracy * 100:.2f}%")

# Calculate improvement
improvement = binary_accuracy - multiclass_accuracy
improvement_percent = (improvement / multiclass_accuracy) * 100

print(f"\nAccuracy Difference: {improvement:.4f} ({improvement_percent:.2f}%)")

if binary_accuracy > multiclass_accuracy:
    print("✓ Binary classification performs BETTER than multi-class classification")
    winner = "Binary"
else:
    print("✗ Multi-class classification performs BETTER than binary classification")
    winner = "Multi-class"

print(f"\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)

print(f"Model Architecture:")
print(f"- Hidden layers: 3 (10, 8, 4 neurons)")
print(f"- Activation: ReLU for hidden layers")
print(f"- Multi-class output: 5 neurons with softmax")
print(f"- Binary output: 1 neuron with sigmoid")
print(f"- Optimizer: Adam (lr=0.001)")
print(f"- Epochs: 60, Batch size: 8")

print(f"\nResults Summary:")
print(f"- Multi-class accuracy: {multiclass_accuracy * 100:.2f}%")
print(f"- Binary accuracy: {binary_accuracy * 100:.2f}%")
print(f"- Difference: {abs(improvement) * 100:.2f} percentage points")
print(f"- Better approach: {winner} Classification")

# Training history analysis
print(f"\nTraining Analysis:")
print(f"- Multi-class final train accuracy: {history_multiclass.history['accuracy'][-1]:.4f}")
print(f"- Multi-class final val accuracy: {history_multiclass.history['val_accuracy'][-1]:.4f}")
print(f"- Binary final train accuracy: {history_binary.history['accuracy'][-1]:.4f}")
print(f"- Binary final val accuracy: {history_binary.history['val_accuracy'][-1]:.4f}")

print(f"\n" + "="*60)
print("CONCLUSION")
print("="*60)
print(f"The {winner.lower()} classification approach achieves better performance")
print(f"with an accuracy of {max(multiclass_accuracy, binary_accuracy) * 100:.2f}%")
print(f"compared to {min(multiclass_accuracy, binary_accuracy) * 100:.2f}% for the other approach.")