"""
CNN Model with parameter count
This script creates a CNN model with specified architecture and counts parameters.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def create_cnn_model():
    """
    Create a CNN model with the following architecture:
    - Input shape: (28, 28, 1)
    - Conv2D: 32 filters, kernel size (3, 3), relu activation
    - MaxPooling2D: pool size (2, 2)
    - Flatten layer
    - Dense: 128 units, relu activation
    - Output Dense: 10 units, softmax activation
    """
    model = keras.Sequential([
        # Input layer (implicit with first layer)
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=(28, 28, 1), name='conv2d'),
        
        layers.MaxPooling2D(pool_size=(2, 2), name='maxpool2d'),
        
        layers.Flatten(name='flatten'),
        
        layers.Dense(128, activation='relu', name='dense_128'),
        
        layers.Dense(10, activation='softmax', name='output_dense')
    ])
    
    return model


def count_parameters_detailed(model):
    """
    Count parameters layer by layer with detailed breakdown
    """
    print("\n" + "="*60)
    print("LAYER-BY-LAYER PARAMETER BREAKDOWN")
    print("="*60)
    
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    for i, layer in enumerate(model.layers):
        layer_params = layer.count_params()
        total_params += layer_params
        
        # Count trainable and non-trainable separately
        layer_trainable = sum([tf.size(w).numpy() for w in layer.trainable_weights])
        layer_non_trainable = sum([tf.size(w).numpy() for w in layer.non_trainable_weights])
        
        trainable_params += layer_trainable
        non_trainable_params += layer_non_trainable
        
        print(f"\nLayer: {layer.name} ({layer.__class__.__name__})")
        # Get output shape from model summary
        if hasattr(layer, 'output'):
            print(f"  Output shape: {layer.output.shape}")
        else:
            print(f"  Output shape: Not available")
        
        if layer_params > 0:
            print(f"  Parameters: {layer_params:,}")
            
            # Show weight shapes for layers with parameters
            if hasattr(layer, 'kernel'):
                kernel_shape = layer.kernel.shape
                print(f"  Kernel shape: {kernel_shape}")
                kernel_params = np.prod(kernel_shape)
                print(f"  Kernel parameters: {kernel_params:,}")
            
            if hasattr(layer, 'bias'):
                if layer.use_bias:
                    bias_shape = layer.bias.shape
                    print(f"  Bias shape: {bias_shape}")
                    print(f"  Bias parameters: {bias_shape[0]:,}")
        else:
            print(f"  Parameters: 0 (no trainable parameters)")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    return total_params


def main():
    # Create the model
    print("Creating CNN Model...")
    model = create_cnn_model()
    
    # Display model summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    model.summary()
    
    # Count parameters with detailed breakdown
    total_params = count_parameters_detailed(model)
    
    # Manual calculation verification
    print("\n" + "="*60)
    print("MANUAL CALCULATION VERIFICATION")
    print("="*60)
    
    print("\n1. Conv2D Layer (32 filters, 3x3 kernel, 1 input channel):")
    conv_params = (3 * 3 * 1 * 32) + 32  # (kernel_h * kernel_w * input_channels * filters) + bias
    print(f"   Parameters = (3 × 3 × 1 × 32) + 32 = {conv_params:,}")
    
    print("\n2. MaxPooling2D Layer:")
    print("   Parameters = 0 (no trainable parameters)")
    
    print("\n3. Flatten Layer:")
    print("   Parameters = 0 (no trainable parameters)")
    
    # Calculate the flattened size after Conv2D and MaxPooling2D
    # Input: 28x28x1 -> Conv2D: 26x26x32 -> MaxPooling2D: 13x13x32
    flattened_size = 13 * 13 * 32
    print(f"\n   Note: After Conv2D (28×28 → 26×26) and MaxPooling2D (26×26 → 13×13),")
    print(f"   the flattened size is 13 × 13 × 32 = {flattened_size:,}")
    
    print("\n4. Dense Layer (128 units):")
    dense1_params = (flattened_size * 128) + 128  # (input_size * units) + bias
    print(f"   Parameters = ({flattened_size:,} × 128) + 128 = {dense1_params:,}")
    
    print("\n5. Output Dense Layer (10 units):")
    dense2_params = (128 * 10) + 10  # (input_size * units) + bias
    print(f"   Parameters = (128 × 10) + 10 = {dense2_params:,}")
    
    print("\n" + "="*60)
    manual_total = conv_params + dense1_params + dense2_params
    print(f"TOTAL (Manual Calculation): {manual_total:,} parameters")
    print("="*60)
    
    # Final answer
    print("\n" + "="*60)
    print("FINAL ANSWER")
    print("="*60)
    print(f"The total number of parameters in the CNN model is: {total_params:,}")
    print("="*60)
    
    return model, total_params


if __name__ == "__main__":
    model, total_params = main()