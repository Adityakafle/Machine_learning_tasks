import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image
from io import BytesIO

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

# Reshape data to add a channel dimension (required for convolutional layers)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Define the encoder layer
encoder = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu')
])

# Define the classifier layers for odd and even number classification
odd_even_classifier = models.Sequential([
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')  # Output layer for odd/even number classification
])

# Combine encoder and classifiers for odd/even number classification
odd_even_model = models.Sequential([
    encoder,
    odd_even_classifier
])

# Compile the model
odd_even_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])

# Train the model
odd_even_model.fit(x_train, y_train % 2, epochs=20, validation_data=(x_test, y_test % 2))

# Save the model locally
odd_even_model.save('odd_even_model.h5')


#Till now model is saved.....................You can Comment this code as model is already saved

# Function to predict handwritten digits using the trained model
def predict_handwritten_digit(image_bytes, model_path):
    # Open the image using PIL
    img = Image.open(BytesIO(image_bytes)).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Predict digit using the model
    predicted_digit = model.predict(img_array)
    predicted_digit = np.argmax(predicted_digit)

    return predicted_digit


# Example usage for digit prediction using model 'model1.h5':
with open('Handwritten_image/8.png', 'rb') as f:
    image_bytes = f.read()
predicted_digit = predict_handwritten_digit(image_bytes, 'model1.h5')
print("Predicted Digit:", predicted_digit)

# Example usage for odd/even prediction using model 'odd_even_model.h5':
with open('Handwritten_image/8.png', 'rb') as f:
    image_bytes = f.read()
predicted_odd_even = predict_handwritten_digit(image_bytes, 'odd_even_model.h5')
print("Predicted Odd/Even:", "Even" if predicted_odd_even == 0 else "Odd")


