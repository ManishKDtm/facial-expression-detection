import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model("facial_emotion_model.keras")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("emotion_model.tflite", "wb") as f:
    f.write(tflite_model)