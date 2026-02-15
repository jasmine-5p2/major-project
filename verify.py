"""
MODEL VERIFICATION SCRIPT
=========================

This script checks if your model files are properly saved and can be loaded.
Run this BEFORE deploying to verify everything is correct.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from pathlib import Path

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
print("MODEL VERIFICATION SCRIPT")
print("="*70)
print()

# Required files
REQUIRED_FILES = {
    'model': ['final_lstm_attention_phishing_detector.keras', 'best_lstm_attention_phishing_detector.keras'],
    'tokenizer': 'tokenizer.json',
    'label_classes': 'label_classes.npy',
    'metadata': 'model_metadata.json'
}

print("Step 1: Checking for required files...")
print("-" * 70)

all_files_present = True

# Check model file
model_file = None
for model_filename in REQUIRED_FILES['model']:
    if os.path.exists(model_filename):
        model_file = model_filename
        size_mb = os.path.getsize(model_filename) / (1024 * 1024)
        print(f"✓ Model file found: {model_filename} ({size_mb:.2f} MB)")
        break

if model_file is None:
    print(f"✗ ERROR: No model file found!")
    print(f"  Looking for: {REQUIRED_FILES['model']}")
    all_files_present = False

# Check other files
for file_type, filename in REQUIRED_FILES.items():
    if file_type == 'model':
        continue
        
    if os.path.exists(filename):
        size_kb = os.path.getsize(filename) / 1024
        print(f"✓ {file_type.title()} file found: {filename} ({size_kb:.2f} KB)")
    else:
        print(f"✗ ERROR: {file_type.title()} file not found: {filename}")
        all_files_present = False

if not all_files_present:
    print("\n" + "="*70)
    print("VERIFICATION FAILED: Missing required files")
    print("="*70)
    print("\nPlease ensure you have run the training script completely.")
    exit(1)

print()
print("Step 2: Loading and verifying model...")
print("-" * 70)

try:
    # Load model
    print(f"Loading model from: {model_file}")
    model = load_model(
        model_file,
        custom_objects={'AttentionLayer': AttentionLayer},
        compile=False
    )
    print("✓ Model loaded successfully")
    
    # Display model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Count layers
    print(f"\nModel has {len(model.layers)} layers")
    
    # Check for embedding layer
    embedding_layer = None
    for layer in model.layers:
        if 'embedding' in layer.name.lower():
            embedding_layer = layer
            break
    
    if embedding_layer:
        print(f"✓ Embedding layer found: {embedding_layer.name}")
        print(f"  Input dim: {embedding_layer.input_dim}")
        print(f"  Output dim: {embedding_layer.output_dim}")
        print(f"  Trainable: {embedding_layer.trainable}")
        
        # Check weights
        weights = embedding_layer.get_weights()
        if len(weights) > 0:
            print(f"✓ Embedding weights present: shape {weights[0].shape}")
        else:
            print(f"✗ WARNING: Embedding layer has no weights!")
    else:
        print("✗ WARNING: No embedding layer found in model")
    
except Exception as e:
    print(f"✗ ERROR loading model: {e}")
    print("\nThis is the same error you're seeing in deployment!")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("Step 3: Testing model inference...")
print("-" * 70)

try:
    # Load metadata to get max_len
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    max_len = metadata['max_len']
    print(f"Max sequence length: {max_len}")
    
    # Create dummy input
    dummy_input = np.zeros((1, max_len), dtype=np.int32)
    print(f"Created dummy input: shape {dummy_input.shape}")
    
    # Make prediction
    print("Running inference...")
    predictions = model.predict(dummy_input, verbose=0)
    print(f"✓ Prediction successful!")
    print(f"  Output shape: {predictions.shape}")
    print(f"  Output probabilities: {predictions[0]}")
    print(f"  Sum of probabilities: {predictions[0].sum():.6f} (should be ~1.0)")
    
except Exception as e:
    print(f"✗ ERROR during inference: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print()
print("Step 4: Verifying tokenizer...")
print("-" * 70)

try:
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    
    with open('tokenizer.json', 'r') as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)
    
    print(f"✓ Tokenizer loaded")
    print(f"  Vocabulary size: {len(tokenizer.word_index)}")
    print(f"  Sample words: {list(tokenizer.word_index.items())[:5]}")
    
    # Test tokenization
    test_url = "example.com/phishing"
    sequences = tokenizer.texts_to_sequences([test_url])
    print(f"\n✓ Test tokenization successful")
    print(f"  Input: {test_url}")
    print(f"  Tokens: {sequences[0][:10]}...")
    
except Exception as e:
    print(f"✗ ERROR loading tokenizer: {e}")
    exit(1)

print()
print("Step 5: Verifying label classes...")
print("-" * 70)

try:
    label_classes = np.load('label_classes.npy', allow_pickle=True)
    print(f"✓ Label classes loaded")
    print(f"  Number of classes: {len(label_classes)}")
    print(f"  Classes: {list(label_classes)}")
    
except Exception as e:
    print(f"✗ ERROR loading label classes: {e}")
    exit(1)

print()
print("="*70)
print("VERIFICATION COMPLETE - ALL CHECKS PASSED! ✅")
print("="*70)
print()
print("Your model is ready for deployment!")
print()
print("Files to include in your repository:")
for file_type, filename in REQUIRED_FILES.items():
    if file_type == 'model':
        print(f"  ✓ {model_file}")
    else:
        print(f"  ✓ {filename}")

print()
print("Next steps:")
print("  1. Replace app.py with app_fixed.py")
print("  2. Ensure all files above are in your git repository")
print("  3. Commit and push to trigger redeployment")
print("  4. Check Render logs to confirm successful startup")