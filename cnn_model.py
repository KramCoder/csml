import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_cnn_model():
    """
    Create a basic CNN model with the specified architecture:
    - Input shape: (28, 28, 1)
    - Conv2D: 32 filters, (3,3) kernel, relu activation
    - MaxPooling2D: (2,2) pool size
    - Flatten layer
    - Dense: 128 units, relu activation
    - Output: 10 units, softmax activation
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(28, 28, 1)),
        
        # 2D Convolutional layer
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        
        # MaxPooling2D layer
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten layer
        layers.Flatten(),
        
        # Dense layer
        layers.Dense(128, activation='relu'),
        
        # Output layer
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def calculate_parameters():
    """
    Create the model and calculate the total number of parameters
    """
    # Create the model
    model = create_cnn_model()
    
    # Compile the model (optional, just for completeness)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Print model summary
    print("CNN Model Architecture:")
    print("=" * 50)
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    
    print("\n" + "=" * 50)
    print(f"Total number of parameters: {total_params:,}")
    print("=" * 50)
    
    return model, total_params

def detailed_parameter_breakdown():
    """
    Provide a detailed breakdown of parameters for each layer
    """
    model = create_cnn_model()
    
    print("\nDetailed Parameter Breakdown:")
    print("=" * 50)
    
    # Get input shape for the first layer
    input_shape = (28, 28, 1)
    
    for i, layer in enumerate(model.layers):
        layer_name = layer.name
        layer_type = layer.__class__.__name__
        
        if hasattr(layer, 'count_params'):
            params = layer.count_params()
            print(f"Layer {i+1}: {layer_name} ({layer_type})")
            print(f"  Parameters: {params:,}")
            
            # Additional details for specific layer types
            if isinstance(layer, layers.Conv2D):
                filters = layer.filters
                kernel_size = layer.kernel_size
                # For the first layer, use the input shape
                if i == 0:
                    input_channels = input_shape[-1]
                else:
                    # For subsequent layers, we need to calculate based on previous layer output
                    # This is a simplified approach - in practice you'd track the output shapes
                    input_channels = 32  # From the Conv2D layer output
                
                # Conv2D parameters = (kernel_height * kernel_width * input_channels + 1) * filters
                # +1 for bias term
                conv_params = (kernel_size[0] * kernel_size[1] * input_channels + 1) * filters
                print(f"  Calculation: ({kernel_size[0]}×{kernel_size[1]}×{input_channels} + 1) × {filters} = {conv_params:,}")
                
            elif isinstance(layer, layers.Dense):
                # For Dense layers, we need to calculate the input units based on previous layer
                if i == 3:  # First Dense layer (after Flatten)
                    # After MaxPooling2D: (13, 13, 32) -> Flatten -> 13*13*32 = 5408
                    input_units = 13 * 13 * 32
                else:  # Output Dense layer
                    input_units = 128
                
                output_units = layer.units
                # Dense parameters = (input_units + 1) * output_units
                # +1 for bias term
                dense_params = (input_units + 1) * output_units
                print(f"  Calculation: ({input_units} + 1) × {output_units} = {dense_params:,}")
            
            print()

if __name__ == "__main__":
    # Create model and calculate parameters
    model, total_params = calculate_parameters()
    
    # Show detailed breakdown
    detailed_parameter_breakdown()
    
    # Additional model information
    print("\nModel Information:")
    print("=" * 50)
    print(f"Model type: {type(model).__name__}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")