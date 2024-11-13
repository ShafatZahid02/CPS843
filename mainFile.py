import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer
import tf2onnx

# Custom layer wrapping the I3D model from TensorFlow Hub
class I3DModelLayer(Layer):
    def __init__(self, **kwargs):
        super(I3DModelLayer, self).__init__(**kwargs)
        # Load the I3D model from TensorFlow Hub
        self.i3d_layer = hub.KerasLayer("https://tfhub.dev/deepmind/i3d-kinetics-400/1", trainable=False)

    def call(self, inputs):
        # I3D model expects inputs in (batch_size, frames, height, width, channels)
        return self.i3d_layer(inputs)

# Define model input shape: (None frames, 224 height, 224 width, 3 color channels)
inputs = tf.keras.Input(shape=(None, 224, 224, 3))
i3d_custom_layer = I3DModelLayer()(inputs)
model = tf.keras.Model(inputs=inputs, outputs=i3d_custom_layer)

# Save model in Keras format
saved_model_dir = "saved_i3d_model.keras"
model.save(saved_model_dir)

# Convert model to ONNX format
onnx_model_path = "saved_i3d_model.onnx"
spec = (tf.TensorSpec((None, None, 224, 224, 3), tf.float32),)

# ONNX conversion
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=onnx_model_path)
print(f"ONNX model saved at: {onnx_model_path}")
