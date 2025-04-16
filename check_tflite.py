import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="emotion_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test with random input (simulates a 48x48 grayscale image)
input_data = np.random.rand(1, 48, 48, 1).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

print("Output shape:", output.shape)  # Should be (1, 7)
print("Sample prediction:", output)