import tensorflow as tf
import numpy as np

# Function to predict handwritten digits using the saved model
def predict_handwritten_digit(image_path, model_path):
    # Load the saved model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the handwritten image
    handwritten_image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(28, 28))
    handwritten_image = tf.keras.preprocessing.image.img_to_array(handwritten_image)
    handwritten_image = handwritten_image / 255.0
    handwritten_image = np.expand_dims(handwritten_image, axis=0)

    # Predict digit using the model
    predicted_digit = model.predict(handwritten_image)
    predicted_digit = np.argmax(predicted_digit)

    return predicted_digit

# Example usage:
image_path = 'Handwritten_image/8.png'
model_path = 'model2.h5'
predicted_digit = predict_handwritten_digit(image_path, model_path)
print("Predicted Digit (Model 2):", predicted_digit)
