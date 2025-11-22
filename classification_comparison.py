#!/usr/bin/env python3
"""
Heart Disease Classification Comparison:
- Multi-class classification (5 classes) with softmax
- Binary classification (2 classes) with sigmoid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import sys
import random
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(19)
np.random.seed(19)
tf.random.set_seed(19)

print("="*60)
print("HEART DISEASE CLASSIFICATION COMPARISON")
print("="*60)
print(f'Python version: {sys.version}')
print(f'TensorFlow version: {tf.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Pandas version: {pd.__version__}')
print("="*60)

# Load the heart disease dataset
dataset = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','class']

# Read the CSV
df = pd.read_csv(dataset, names=column_names)

print("\nDataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Data preprocessing
df = df.replace('?', np.nan)
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['thal'] = pd.to_numeric(df['thal'], errors='coerce')
df = df.dropna()

# Check class distribution
print("\nOriginal Class Distribution:")
print(df['class'].value_counts().sort_index())

# Separate features and target
X = df.drop('class', axis=1).values
y = df['class'].values

# Store original multi-class labels
y_multi = y.copy()

# Convert to binary classification (0: no disease, 1-4: disease present)
y_binary = (y > 0).astype(int)

print("\nBinary Class Distribution:")
print(f"No Disease (0): {np.sum(y_binary == 0)}")
print(f"Disease Present (1): {np.sum(y_binary == 1)}")

# Split data (using same split for both models)
X_train, X_test, y_train_multi, y_test_multi = train_test_split(
    X, y_multi, test_size=0.2, shuffle=False, random_state=90
)

# Binary labels using the same split
y_train_binary = (y_train_multi > 0).astype(int)
y_test_binary = (y_test_multi > 0).astype(int)

# One-hot encode multi-class labels
y_train_multi_cat = to_categorical(y_train_multi, num_classes=5)
y_test_multi_cat = to_categorical(y_test_multi, num_classes=5)

print("\nData Split:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print(f"Input features: {X_train.shape[1]}")

# Common hyperparameters (exact same as original)
EPOCHS = 60
BATCH_SIZE = 8
LEARNING_RATE = 0.001

def create_multi_class_model(input_dim):
    """Create the original multi-class classification model"""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(5, activation='softmax'))  # 5 classes with softmax
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )
    return model

def create_binary_class_model(input_dim):
    """Create the binary classification model with sigmoid"""
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary output with sigmoid
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )
    return model

print("\n" + "="*60)
print("TRAINING MULTI-CLASS CLASSIFICATION MODEL")
print("="*60)

# Create and train multi-class model
model_multi = create_multi_class_model(X_train.shape[1])
print("\nMulti-class Model Architecture:")
model_multi.summary()

# Callbacks for multi-class
filepath_multi = "multi_class_model-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint_multi = ModelCheckpoint(
    filepath_multi, 
    monitor='val_accuracy', 
    verbose=0, 
    save_best_only=True, 
    mode='max'
)
early_stop = EarlyStopping(monitor='val_accuracy', patience=15, verbose=0)

# Train multi-class model
history_multi = model_multi.fit(
    X_train, y_train_multi_cat,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    validation_data=(X_test, y_test_multi_cat),
    callbacks=[checkpoint_multi, early_stop]
)

# Evaluate multi-class model
y_pred_multi_proba = model_multi.predict(X_test, verbose=0)
y_pred_multi = np.argmax(y_pred_multi_proba, axis=1)
y_test_multi_argmax = np.argmax(y_test_multi_cat, axis=1)

multi_class_accuracy = accuracy_score(y_test_multi_argmax, y_pred_multi)
print(f"\nMulti-class Classification Results:")
print(f"Test Accuracy: {multi_class_accuracy * 100:.2f}%")
print(f"Best Validation Accuracy: {max(history_multi.history['val_accuracy']) * 100:.2f}%")
print(f"Final Training Accuracy: {history_multi.history['accuracy'][-1] * 100:.2f}%")

print("\n" + "="*60)
print("TRAINING BINARY CLASSIFICATION MODEL")
print("="*60)

# Create and train binary model
model_binary = create_binary_class_model(X_train.shape[1])
print("\nBinary Model Architecture:")
model_binary.summary()

# Callbacks for binary
filepath_binary = "binary_model-{epoch:02d}-{val_accuracy:.2f}.h5"
checkpoint_binary = ModelCheckpoint(
    filepath_binary,
    monitor='val_accuracy',
    verbose=0,
    save_best_only=True,
    mode='max'
)
early_stop_binary = EarlyStopping(monitor='val_accuracy', patience=15, verbose=0)

# Train binary model
history_binary = model_binary.fit(
    X_train, y_train_binary,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    validation_data=(X_test, y_test_binary),
    callbacks=[checkpoint_binary, early_stop_binary]
)

# Evaluate binary model
y_pred_binary_proba = model_binary.predict(X_test, verbose=0)
y_pred_binary = (y_pred_binary_proba > 0.5).astype(int).flatten()

binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"\nBinary Classification Results:")
print(f"Test Accuracy: {binary_accuracy * 100:.2f}%")
print(f"Best Validation Accuracy: {max(history_binary.history['val_accuracy']) * 100:.2f}%")
print(f"Final Training Accuracy: {history_binary.history['accuracy'][-1] * 100:.2f}%")

# Comparison and Analysis
print("\n" + "="*60)
print("COMPARISON AND ANALYSIS")
print("="*60)

print(f"\n📊 ACCURACY COMPARISON:")
print(f"   Multi-class (5 classes, softmax): {multi_class_accuracy * 100:.2f}%")
print(f"   Binary (2 classes, sigmoid):      {binary_accuracy * 100:.2f}%")
print(f"   Improvement:                       {(binary_accuracy - multi_class_accuracy) * 100:+.2f}%")

if binary_accuracy > multi_class_accuracy:
    print(f"\n✅ Binary classification performs BETTER by {(binary_accuracy - multi_class_accuracy) * 100:.2f}%")
    better_model = "Binary"
elif binary_accuracy < multi_class_accuracy:
    print(f"\n✅ Multi-class classification performs BETTER by {(multi_class_accuracy - binary_accuracy) * 100:.2f}%")
    better_model = "Multi-class"
else:
    print(f"\n➡️  Both models have the SAME accuracy")
    better_model = "Both equal"

# Detailed classification reports
print("\n" + "-"*40)
print("MULTI-CLASS CLASSIFICATION REPORT:")
print("-"*40)
print(classification_report(y_test_multi_argmax, y_pred_multi, target_names=[f'Class {i}' for i in range(5)]))

print("\n" + "-"*40)
print("BINARY CLASSIFICATION REPORT:")
print("-"*40)
print(classification_report(y_test_binary, y_pred_binary, target_names=['No Disease', 'Disease Present']))

# Plot training history comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Training History Comparison: Multi-class vs Binary Classification', fontsize=16)

# Multi-class accuracy
axes[0, 0].plot(history_multi.history['accuracy'], label='Training', linewidth=2)
axes[0, 0].plot(history_multi.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_title('Multi-class Model Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Multi-class loss
axes[0, 1].plot(history_multi.history['loss'], label='Training', linewidth=2)
axes[0, 1].plot(history_multi.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_title('Multi-class Model Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Binary accuracy
axes[1, 0].plot(history_binary.history['accuracy'], label='Training', linewidth=2)
axes[1, 0].plot(history_binary.history['val_accuracy'], label='Validation', linewidth=2)
axes[1, 0].set_title('Binary Model Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Binary loss
axes[1, 1].plot(history_binary.history['loss'], label='Training', linewidth=2)
axes[1, 1].plot(history_binary.history['val_loss'], label='Validation', linewidth=2)
axes[1, 1].set_title('Binary Model Loss')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Loss')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_comparison.png', dpi=100)
print("\n📈 Training history plot saved as 'training_history_comparison.png'")

# Create confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Confusion Matrices Comparison', fontsize=16)

# Multi-class confusion matrix
cm_multi = confusion_matrix(y_test_multi_argmax, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'Multi-class Confusion Matrix\n(Accuracy: {multi_class_accuracy * 100:.2f}%)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Binary confusion matrix
cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'Binary Confusion Matrix\n(Accuracy: {binary_accuracy * 100:.2f}%)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100)
print("📊 Confusion matrices saved as 'confusion_matrices.png'")

# Analysis Summary
print("\n" + "="*60)
print("ANALYSIS SUMMARY")
print("="*60)

print("\n1. MODEL COMPLEXITY:")
print(f"   - Multi-class: Predicts 5 distinct severity levels (0-4)")
print(f"   - Binary: Simplified to disease present/absent")

print("\n2. PERFORMANCE INSIGHTS:")
if binary_accuracy > multi_class_accuracy:
    print(f"   - Binary classification is {(binary_accuracy - multi_class_accuracy) * 100:.2f}% more accurate")
    print(f"   - Simpler problem formulation leads to better performance")
    print(f"   - Binary distinction (disease/no disease) is more reliable")
else:
    print(f"   - Multi-class maintains granular disease severity information")
    print(f"   - Performance difference: {(multi_class_accuracy - binary_accuracy) * 100:.2f}%")

print("\n3. PRACTICAL IMPLICATIONS:")
print(f"   - Binary model: Better for screening (disease yes/no)")
print(f"   - Multi-class model: Better for severity assessment")
print(f"   - Choice depends on clinical application needs")

print("\n4. RECOMMENDATION:")
if binary_accuracy - multi_class_accuracy > 0.1:
    print(f"   ✅ Use BINARY classification for this dataset")
    print(f"      Reason: Significantly higher accuracy ({binary_accuracy * 100:.2f}% vs {multi_class_accuracy * 100:.2f}%)")
elif multi_class_accuracy - binary_accuracy > 0.1:
    print(f"   ✅ Use MULTI-CLASS classification for this dataset")
    print(f"      Reason: Better accuracy with severity information")
else:
    print(f"   ➡️  Both models perform similarly")
    print(f"      Choose based on application requirements")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)