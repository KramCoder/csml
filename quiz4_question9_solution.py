#!/usr/bin/env python3
"""
Quiz 4 Question 9 - CNN with Modified Architecture

This script implements a CNN with the following modifications:
1. Convolutional layer with 32 3x3 filters (instead of 16), using valid padding, followed by ReLU activation
2. Fully-connected layer with 150 units (instead of 100) and ReLU activation
3. Training for 20 epochs instead of 10

The answer to whether overfitting has been resolved: FALSE
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def create_modified_cnn():
    """
    Create the modified CNN model with:
    - 32 3x3 conv filters (valid padding) instead of 16
    - 150 dense units instead of 100
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(28, 28, 1)),
        
        # MODIFICATION 1: 32 filters instead of 16
        layers.Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu'),
        
        # Max pooling layer
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten
        layers.Flatten(),
        
        # MODIFICATION 2: 150 units instead of 100
        layers.Dense(150, activation='relu'),
        
        # Output layer
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    # Load data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def analyze_overfitting(history):
    """Analyze whether overfitting has been resolved"""
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*50)
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    loss_gap = final_val_loss - final_train_loss
    acc_gap = final_train_acc - final_val_acc
    
    print(f"\nLoss Gap (Val - Train): {loss_gap:.4f}")
    print(f"Accuracy Gap (Train - Val): {acc_gap:.4f}")
    
    # Determine if overfitting is resolved
    overfitting_resolved = loss_gap <= 0.1 and acc_gap <= 0.05
    
    return overfitting_resolved

def main():
    print("Quiz 4 Question 9 - CNN Solution")
    print("="*50)
    
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    print("\nLoading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Create model
    print("\nCreating modified CNN model...")
    model = create_modified_cnn()
    model.summary()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    # MODIFICATION 3: Train for 20 epochs instead of 10
    print("\nTraining model for 20 epochs...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=20,  # Changed from 10 to 20
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Analyze overfitting
    overfitting_resolved = analyze_overfitting(history)
    
    # Print the quiz answer
    print("\n" + "="*50)
    print("QUIZ ANSWER: Has the overfitting issue been resolved?")
    print("="*50)
    print(f"Answer: {'True' if overfitting_resolved else 'False'}")
    print("\nExplanation:")
    print("The modifications made to the model:")
    print("  1. Increased conv filters: 16 → 32 (increases capacity)")
    print("  2. Increased dense units: 100 → 150 (increases capacity)")
    print("  3. Increased epochs: 10 → 20 (more training time)")
    print("\nAll these changes INCREASE model capacity and training time,")
    print("which typically WORSENS overfitting rather than resolving it.")
    print("\nExpected Answer: FALSE - Overfitting is NOT resolved")
    print("="*50)
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add annotation for overfitting region
    if len(history.history['loss']) > 5:
        mid_point = len(history.history['loss']) // 2
        if history.history['val_loss'][mid_point] > history.history['loss'][mid_point]:
            ax1.annotate('Overfitting starts here',
                        xy=(mid_point, history.history['val_loss'][mid_point]),
                        xytext=(mid_point+2, history.history['val_loss'][mid_point]+0.1),
                        arrowprops=dict(arrowstyle='->', color='red'))
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training vs Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Overfitting Analysis - Quiz 4 Question 9', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return overfitting_resolved

if __name__ == "__main__":
    result = main()