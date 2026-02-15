"""
CONVERT MODEL TO WEIGHTS-ONLY FORMAT
=====================================

This script rebuilds and saves your model in a format that works across
different environments, avoiding the TensorFlow 2.14/2.15 serialization bug.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super().build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        out = x * a
        return K.sum(out, axis=1)
    
    def get_config(self):
        return super().get_config()

print("="*70)
print("MODEL CONVERSION SCRIPT")
print("="*70)
print()

# Load metadata
print("Step 1: Loading metadata...")
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

MAX_WORDS = metadata.get('max_words', 15000)
MAX_LEN = metadata.get('max_len', 120)
EMBED_DIM = metadata.get('embed_dim', 128)
LSTM_UNITS = metadata.get('lstm_units', 96)
DROPOUT_RATE = 0.3
num_classes = metadata['num_classes']

print(f"  Max Words: {MAX_WORDS}")
print(f"  Max Length: {MAX_LEN}")
print(f"  Embed Dim: {EMBED_DIM}")
print(f"  LSTM Units: {LSTM_UNITS}")
print(f"  Num Classes: {num_classes}")
print()

# Load original model to extract weights
print("Step 2: Loading original model...")
try:
    original_model = load_model(
        'final_lstm_attention_phishing_detector.keras',
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    print("✓ Original model loaded")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    print("\nTrying to load from best model...")
    try:
        original_model = load_model(
            'best_lstm_attention_phishing_detector.keras',
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print("✓ Loaded from best model")
    except Exception as e2:
        print(f"✗ Error: {e2}")
        exit(1)

print()

# Extract weights
print("Step 3: Extracting weights...")
weights = original_model.get_weights()
print(f"✓ Extracted {len(weights)} weight arrays")
for i, w in enumerate(weights):
    print(f"  Weight {i}: shape {w.shape}")
print()

# Build new model with same architecture
print("Step 4: Building fresh model...")
inputs = Input(shape=(MAX_LEN,), name='input')
embedding = Embedding(MAX_WORDS, EMBED_DIM, name='embedding')(inputs)
lstm_out = LSTM(LSTM_UNITS, return_sequences=True, name='lstm')(embedding)
attention_out = AttentionLayer(name='attention')(lstm_out)
dropout = Dropout(DROPOUT_RATE, name='dropout')(attention_out)
dense = Dense(64, activation="relu", name='dense')(dropout)
outputs = Dense(num_classes, activation="softmax", name='output')(dense)

new_model = Model(inputs, outputs, name='LSTM_Attention_Phishing_Detector')
print("✓ Model architecture built")
print()

# Set weights
print("Step 5: Setting weights...")
new_model.set_weights(weights)
print("✓ Weights set successfully")
print()

# Compile model
print("Step 6: Compiling model...")
new_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=["accuracy"]
)
print("✓ Model compiled")
print()

# Test inference
print("Step 7: Testing inference...")
dummy_input = np.zeros((1, MAX_LEN), dtype=np.int32)
predictions = new_model.predict(dummy_input, verbose=0)
print(f"✓ Inference test successful: output shape {predictions.shape}")
print()

# Save model architecture as JSON
print("Step 8: Saving model architecture...")
model_json = new_model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)
print("✓ Saved model architecture to 'model_architecture.json'")
print()

# Save weights only
print("Step 9: Saving weights...")
new_model.save_weights('model_weights.h5')
print("✓ Saved weights to 'model_weights.h5'")
print()

# Also save in Keras 3 format for backup
print("Step 10: Saving as backup .keras file...")
new_model.save('model_deployment.keras')
print("✓ Saved backup to 'model_deployment.keras'")
print()

print("="*70)
print("CONVERSION COMPLETE!")
print("="*70)
print()
print("Generated files for deployment:")
print("  ✓ model_architecture.json  - Model architecture")
print("  ✓ model_weights.h5         - Model weights (USE THIS)")
print("  ✓ model_deployment.keras   - Backup full model")
print()
print("Next steps:")
print("  1. Replace your app.py with app_weights_loading.py")
print("  2. Commit these files:")
print("     - model_architecture.json")
print("     - model_weights.h5")
print("     - tokenizer.json")
print("     - label_classes.npy")
print("     - model_metadata.json")
print("     - app.py (updated version)")
print("  3. Push to GitHub")
print()