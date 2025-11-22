import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import random
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

print("keras version:", tf.keras.__version__)
print("python version:", sys.version)
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("tensorflow version:", tf.__version__)

# Import the heart disease dataset
dataset = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

column_names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','class']

# Read the csv
dataset = pd.read_csv(dataset, names=column_names)

# Replace '?' with NaN
dataset = dataset.replace('?', np.nan)

# Convert to numeric, errors='coerce' will convert non-numeric to NaN
dataset = dataset.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
data = dataset.dropna()

print("Data shape:", data.shape)
print("Data types:")
print(data.dtypes)

# Create X and Y datasets for training
X = data.iloc[:,0:13]
y = data.iloc[:,-1]

print("\nOriginal class distribution:")
print(y.value_counts().sort_index())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=90)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# =============================================================================
# MULTI-CLASS CLASSIFICATION MODEL
# =============================================================================
print("\n" + "="*60)
print("MULTI-CLASS CLASSIFICATION MODEL")
print("="*60)

# Prepare data for multi-class classification
y_train_multiclass = to_categorical(y_train)
y_test_multiclass = to_categorical(y_test)

print(f"Multi-class training labels shape: {y_train_multiclass.shape}")
print(f"Multi-class test labels shape: {y_test_multiclass.shape}")

# Create multi-class model
model_multiclass = Sequential()
model_multiclass.add(Input(shape=(13,)))
model_multiclass.add(Dense(10, kernel_initializer='normal', activation='relu'))
model_multiclass.add(Dense(8, kernel_initializer='normal', activation='relu'))
model_multiclass.add(Dense(4, kernel_initializer='normal', activation='relu'))
model_multiclass.add(Dense(5, activation='softmax'))

# Compile multi-class model
model_multiclass.compile(loss='categorical_crossentropy', 
                        optimizer=Adam(learning_rate=0.001), 
                        metrics=['accuracy'])

print("\nMulti-class model summary:")
model_multiclass.summary()

# Train multi-class model
print("\nTraining multi-class model...")
history_multiclass = model_multiclass.fit(X_train, y_train_multiclass, 
                                         epochs=60, batch_size=8, verbose=1, 
                                         validation_data=(X_test, y_test_multiclass))

# Evaluate multi-class model
pred_multiclass = model_multiclass.predict(X_test)
y_pred_multiclass = np.argmax(pred_multiclass, axis=1)
y_test_multiclass_labels = np.argmax(y_test_multiclass, axis=1)

multiclass_accuracy = accuracy_score(y_test_multiclass_labels, y_pred_multiclass)
print(f"\nMulti-class Classification Accuracy: {multiclass_accuracy * 100:.2f}%")

# =============================================================================
# BINARY CLASSIFICATION MODEL
# =============================================================================
print("\n" + "="*60)
print("BINARY CLASSIFICATION MODEL")
print("="*60)

# Convert to binary classification: 0 = no disease, 1 = disease (any level)
y_binary = (y > 0).astype(int)
y_train_binary = y_binary[y_train.index]
y_test_binary = y_binary[y_test.index]

print("Binary class distribution:")
print(f"No disease (0): {np.sum(y_binary == 0)}")
print(f"Disease (1): {np.sum(y_binary == 1)}")

print(f"\nBinary training labels shape: {y_train_binary.shape}")
print(f"Binary test labels shape: {y_test_binary.shape}")

# Create binary model
model_binary = Sequential()
model_binary.add(Input(shape=(13,)))
model_binary.add(Dense(10, kernel_initializer='normal', activation='relu'))
model_binary.add(Dense(8, kernel_initializer='normal', activation='relu'))
model_binary.add(Dense(4, kernel_initializer='normal', activation='relu'))
model_binary.add(Dense(1, activation='sigmoid'))  # Binary classification with sigmoid

# Compile binary model
model_binary.compile(loss='binary_crossentropy', 
                    optimizer=Adam(learning_rate=0.001), 
                    metrics=['accuracy'])

print("\nBinary model summary:")
model_binary.summary()

# Train binary model
print("\nTraining binary model...")
history_binary = model_binary.fit(X_train, y_train_binary, 
                                 epochs=60, batch_size=8, verbose=1, 
                                 validation_data=(X_test, y_test_binary))

# Evaluate binary model
pred_binary = model_binary.predict(X_test)
y_pred_binary = (pred_binary > 0.5).astype(int).flatten()

binary_accuracy = accuracy_score(y_test_binary, y_pred_binary)
print(f"\nBinary Classification Accuracy: {binary_accuracy * 100:.2f}%")

# =============================================================================
# COMPARISON AND ANALYSIS
# =============================================================================
print("\n" + "="*60)
print("COMPARISON AND ANALYSIS")
print("="*60)

print(f"\nMulti-class Classification Accuracy: {multiclass_accuracy * 100:.2f}%")
print(f"Binary Classification Accuracy: {binary_accuracy * 100:.2f}%")

# Calculate improvement
improvement = binary_accuracy - multiclass_accuracy
improvement_percent = (improvement / multiclass_accuracy) * 100

print(f"\nAccuracy Improvement: {improvement:.4f} ({improvement_percent:.2f}%)")

if binary_accuracy > multiclass_accuracy:
    print("✓ Binary classification performs BETTER than multi-class classification")
else:
    print("✗ Multi-class classification performs BETTER than binary classification")

# Detailed analysis
print(f"\nDetailed Analysis:")
print(f"- Multi-class model: {multiclass_accuracy * 100:.2f}% accuracy")
print(f"- Binary model: {binary_accuracy * 100:.2f}% accuracy")
print(f"- Difference: {abs(improvement) * 100:.2f} percentage points")

# Confusion matrices
print(f"\nMulti-class Confusion Matrix:")
cm_multiclass = confusion_matrix(y_test_multiclass_labels, y_pred_multiclass)
print(cm_multiclass)

print(f"\nBinary Confusion Matrix:")
cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
print(cm_binary)

# Plotting comparison
plt.figure(figsize=(15, 10))

# Accuracy comparison
plt.subplot(2, 3, 1)
plt.plot(history_multiclass.history['accuracy'], label='Multi-class Train', marker='.')
plt.plot(history_multiclass.history['val_accuracy'], label='Multi-class Val', marker='.')
plt.plot(history_binary.history['accuracy'], label='Binary Train', marker='.')
plt.plot(history_binary.history['val_accuracy'], label='Binary Val', marker='.')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss comparison
plt.subplot(2, 3, 2)
plt.plot(history_multiclass.history['loss'], label='Multi-class Train', marker='.')
plt.plot(history_multiclass.history['val_loss'], label='Multi-class Val', marker='.')
plt.plot(history_binary.history['loss'], label='Binary Train', marker='.')
plt.plot(history_binary.history['val_loss'], label='Binary Val', marker='.')
plt.title('Model Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Final accuracy bar chart
plt.subplot(2, 3, 3)
models = ['Multi-class', 'Binary']
accuracies = [multiclass_accuracy * 100, binary_accuracy * 100]
colors = ['skyblue', 'lightcoral']
bars = plt.bar(models, accuracies, color=colors)
plt.title('Final Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.2f}%', ha='center', va='bottom')

plt.grid(True, alpha=0.3)

# Multi-class confusion matrix heatmap
plt.subplot(2, 3, 4)
plt.imshow(cm_multiclass, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Multi-class Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(5)
plt.xticks(tick_marks, [f'Class {i}' for i in range(5)])
plt.yticks(tick_marks, [f'Class {i}' for i in range(5)])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add text annotations
thresh = cm_multiclass.max() / 2.
for i, j in np.ndindex(cm_multiclass.shape):
    plt.text(j, i, format(cm_multiclass[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm_multiclass[i, j] > thresh else "black")

# Binary confusion matrix heatmap
plt.subplot(2, 3, 5)
plt.imshow(cm_binary, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Binary Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Disease', 'Disease'])
plt.yticks(tick_marks, ['No Disease', 'Disease'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Add text annotations
thresh = cm_binary.max() / 2.
for i, j in np.ndindex(cm_binary.shape):
    plt.text(j, i, format(cm_binary[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm_binary[i, j] > thresh else "black")

# Class distribution
plt.subplot(2, 3, 6)
original_classes = y.value_counts().sort_index()
binary_classes = [np.sum(y_binary == 0), np.sum(y_binary == 1)]
x_pos = np.arange(len(original_classes))
plt.bar(x_pos, original_classes.values, alpha=0.7, label='Original Classes')
plt.bar([0, 1], binary_classes, alpha=0.7, color='red', label='Binary Classes')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.legend()
plt.xticks(range(5), [f'Class {i}' for i in range(5)])

plt.tight_layout()
plt.savefig('binary_vs_multiclass_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Summary report
print(f"\n" + "="*60)
print("SUMMARY REPORT")
print("="*60)
print(f"Dataset: Heart Disease (Cleveland)")
print(f"Features: 13")
print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"Original classes: 5 (0-4)")
print(f"Binary classes: 2 (0=No Disease, 1=Disease)")
print(f"\nModel Architecture:")
print(f"- Hidden layers: 3 (10, 8, 4 neurons)")
print(f"- Activation: ReLU")
print(f"- Optimizer: Adam (lr=0.001)")
print(f"- Epochs: 60")
print(f"- Batch size: 8")
print(f"\nResults:")
print(f"- Multi-class accuracy: {multiclass_accuracy * 100:.2f}%")
print(f"- Binary accuracy: {binary_accuracy * 100:.2f}%")
print(f"- Improvement: {improvement * 100:.2f} percentage points")
print(f"- Better approach: {'Binary' if binary_accuracy > multiclass_accuracy else 'Multi-class'}")