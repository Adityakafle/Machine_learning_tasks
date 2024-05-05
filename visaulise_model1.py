import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('model1.h5')

# Display model summary to identify the layer you want to visualize
print(model.summary())

# Extract feature maps for a sample of input images
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0  # Normalize pixel values to [0, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
sample_images = x_train[:10]

# Get the output of the desired layer
layer_name = 'conv2d'  # Change this to the name of the layer you want to visualize
intermediate_layer_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
feature_maps = intermediate_layer_model.predict(sample_images)

# Visualize feature maps
def visualize_feature_maps(feature_maps):
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(feature_maps[i, :, :, 0], cmap='viridis')
        plt.title(f'Feature Map {i+1}')
        plt.axis('off')
    plt.suptitle(f'Feature Maps - Layer: {layer_name}')
    plt.show()

visualize_feature_maps(feature_maps)
