#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score
import sys
import random
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

print("Heart Disease Prediction: Multi-class vs Binary Classification Comparison")
print("="*80)

print(f'Python version: {sys.version}')
print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
print(f'TensorFlow version: {tf.__version__}')

# Set random seeds for reproducibility
random.seed(19)
np.random.seed(19)
tf.random.set_seed(19)

print("\n" + "="*80)
print("DATA LOADING AND PREPROCESSING")
print("="*80)

# Load the heart disease dataset
column_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', names=column_names)

print("Original dataset info:")
print(f"Dataset shape: {df.shape}")

# Replace '?' with NaN and then drop rows with NaN values
df = df.replace('?', np.nan)
df = df.dropna(axis=0)
data = df.apply(pd.to_numeric)

print(f"After preprocessing shape: {data.shape}")

# Analyze target classes
print("\nOriginal Multi-class Distribution:")
print(data['class'].value_counts().sort_index())

# Create features and targets
X = data.iloc[:,0:13]
y_multi = data.iloc[:,-1]  # Multi-class target

# Create binary classification target
y_binary = (y_multi > 0).astype(int)

print("\nBinary Classification Distribution:")
print(y_binary.value_counts().sort_index())
print("Binary classes mapping:")
print("0: No heart disease")
print("1: Heart disease (any level)")

print("\n" + "="*80)
print("MULTI-CLASS CLASSIFICATION MODEL")
print("="*80)

# Split data for multi-class classification
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X, y_multi, test_size=0.2, shuffle=False, random_state=90)

# One-hot encode for multi-class (5 classes: 0, 1, 2, 3, 4)
y_train_multi_cat = to_categorical(y_train_multi, num_classes=5)
y_test_multi_cat = to_categorical(y_test_multi, num_classes=5)

print(f"Multi-class training shapes: X={X_train_multi.shape}, y={y_train_multi_cat.shape}")

# Build multi-class model
model_multi = Sequential([
    Input(shape=(13,)),
    Dense(10, kernel_initializer='normal', activation='relu'),
    Dense(8, kernel_initializer='normal', activation='relu'),
    Dense(4, kernel_initializer='normal', activation='relu'),
    Dense(5, activation='softmax')  # 5 classes with softmax
])

print("Multi-class Model Architecture:")
model_multi.summary()

# Compile multi-class model
model_multi.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train multi-class model
print("\nTraining Multi-class Model...")
history_multi = model_multi.fit(
    X_train_multi, y_train_multi_cat, 
    epochs=60, batch_size=8, verbose=0, 
    validation_data=(X_test_multi, y_test_multi_cat)
)

print("\n" + "="*80)
print("BINARY CLASSIFICATION MODEL")
print("="*80)

# Split data for binary classification
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, shuffle=False, random_state=90)

print(f"Binary training shapes: X={X_train_binary.shape}, y={y_train_binary.shape}")
print(f"Binary target distribution in training: {pd.Series(y_train_binary).value_counts().sort_index().to_dict()}")

# Build binary classification model
model_binary = Sequential([
    Input(shape=(13,)),
    Dense(10, kernel_initializer='normal', activation='relu'),
    Dense(8, kernel_initializer='normal', activation='relu'),
    Dense(4, kernel_initializer='normal', activation='relu'),
    Dense(1, activation='sigmoid')  # 1 output with sigmoid for binary classification
])

print("Binary Classification Model Architecture:")
model_binary.summary()

# Compile binary model
model_binary.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Train binary model
print("\nTraining Binary Classification Model...")
history_binary = model_binary.fit(
    X_train_binary, y_train_binary, 
    epochs=60, batch_size=8, verbose=0, 
    validation_data=(X_test_binary, y_test_binary)
)

print("\n" + "="*80)
print("MODEL EVALUATION AND COMPARISON")
print("="*80)

# Evaluate Multi-class Model
print("MULTI-CLASS MODEL EVALUATION")
print("-" * 40)
pred_multi = model_multi.predict(X_test_multi, verbose=0)
y_pred_multi_argmax = np.argmax(pred_multi, axis=1)
y_test_multi_argmax = np.argmax(y_test_multi_cat, axis=1)

multi_accuracy = accuracy_score(y_test_multi_argmax, y_pred_multi_argmax) * 100
print(f'Multi-class Classification Accuracy: {multi_accuracy:.2f}%')

# Evaluate Binary Model
print("\nBINARY CLASSIFICATION MODEL EVALUATION")
print("-" * 40)
pred_binary = model_binary.predict(X_test_binary, verbose=0)
y_pred_binary = (pred_binary > 0.5).astype(int).flatten()

binary_accuracy = accuracy_score(y_test_binary, y_pred_binary) * 100
print(f'Binary Classification Accuracy: {binary_accuracy:.2f}%')

print("\n" + "="*80)
print("FINAL COMPARISON RESULTS")
print("="*80)

print(f"Multi-class Classification Accuracy: {multi_accuracy:.2f}%")
print(f"Binary Classification Accuracy: {binary_accuracy:.2f}%")
print(f"Difference: {binary_accuracy - multi_accuracy:.2f} percentage points")

if binary_accuracy > multi_accuracy:
    print(f"\n🏆 WINNER: Binary Classification performs better!")
    print(f"Binary classification is {binary_accuracy - multi_accuracy:.2f} percentage points more accurate.")
    print("\nPossible reasons for better binary performance:")
    print("• Simpler decision boundary (disease vs no disease)")
    print("• More balanced classes after grouping")
    print("• Reduced complexity eliminates confusion between disease severity levels")
elif multi_accuracy > binary_accuracy:
    print(f"\n🏆 WINNER: Multi-class Classification performs better!")
    print(f"Multi-class classification is {multi_accuracy - binary_accuracy:.2f} percentage points more accurate.")
    print("\nPossible reasons for better multi-class performance:")
    print("• Preserves important information about disease severity")
    print("• Model can learn more nuanced patterns")
    print("• Different disease levels have distinct characteristics")
else:
    print("\n🤝 TIE: Both models perform equally well!")

# Show final training accuracies
print(f"\nFinal Training Accuracies:")
print(f"Multi-class - Training: {history_multi.history['accuracy'][-1]:.4f}, Validation: {history_multi.history['val_accuracy'][-1]:.4f}")
print(f"Binary - Training: {history_binary.history['accuracy'][-1]:.4f}, Validation: {history_binary.history['val_accuracy'][-1]:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)