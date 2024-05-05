import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train / 255.0  # Normalize pixel values to [0, 1]
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension

# Load the pre-trained encoder models
encoder_digit = tf.keras.models.load_model('model1.h5')
encoder_odd_even = tf.keras.models.load_model('odd_even_model.h5')

# Extract feature maps for a sample of input images
sample_images = x_train[:10]  # Choose a sample of input images
feature_maps_digit = encoder_digit.predict(sample_images)
feature_maps_odd_even = encoder_odd_even.predict(sample_images)

# Ensure both feature maps have the same shape
num_filters = min(feature_maps_digit.shape[3], feature_maps_odd_even.shape[3])
feature_maps_digit = feature_maps_digit[:, :, :, :num_filters]
feature_maps_odd_even = feature_maps_odd_even[:, :, :, :num_filters]

# Compare feature maps using metrics
# For example, you can calculate Mean Squared Error (MSE)
mse = np.mean(np.square(feature_maps_digit - feature_maps_odd_even))

# Visualize feature maps
def visualize_feature_maps(feature_maps, title):
    num_images = feature_maps.shape[0]
    num_cols = min(4, num_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 3*num_rows))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(feature_maps[i, :, :, 0], cmap='viridis')
            ax.axis('off')
            ax.set_title(f'Feature Map {i+1}')
        else:
            ax.axis('off')
    fig.suptitle(title)
    plt.show()

visualize_feature_maps(feature_maps_digit, 'Feature Maps - Digit Classifier')
visualize_feature_maps(feature_maps_odd_even, 'Feature Maps - Odd/Even Classifier')

print(f'Mean Squared Error (MSE) between feature maps: {mse}')
