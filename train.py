"""
LSTM + ATTENTION PHISHING DETECTION - COMPLETE TRAINING SCRIPT
================================================================

Train LSTM with Attention mechanism on your latest 4-category URL dataset:
- Benign
- Phishing
- Malware
- Spam

This script handles everything from data loading to model deployment.
"""

import pandas as pd
import numpy as np
import re
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# =============================================
# CONFIGURATION
# =============================================
# UPDATE THIS with your dataset filename
DATASET_FILE = "datasets.csv"  # or "your_dataset.csv"

# Model hyperparameters
MAX_WORDS = 15000      # Vocabulary size
MAX_LEN = 120          # Maximum URL length
EMBED_DIM = 128        # Embedding dimension
LSTM_UNITS = 96        # LSTM units
DROPOUT_RATE = 0.3     # Dropout rate

# Training parameters
EPOCHS = 15            # Number of training epochs
BATCH_SIZE = 32        # Batch size
LEARNING_RATE = 0.001  # Learning rate
TEST_SIZE = 0.2        # Test set proportion
RANDOM_STATE = 42      # Random seed for reproducibility

# Output filenames
MODEL_NAME = "lstm_attention_phishing_detector"
BEST_MODEL = f"best_{MODEL_NAME}.keras"
FINAL_MODEL = f"final_{MODEL_NAME}.keras"
TOKENIZER_FILE = "tokenizer.json"
LABEL_CLASSES_FILE = "label_classes.npy"
TRAINING_LOG = "training_log.csv"
METADATA_FILE = "model_metadata.json"

print("="*70)
print("LSTM + ATTENTION PHISHING DETECTION - TRAINING")
print("="*70)
print(f"\nConfiguration:")
print(f"  Dataset: {DATASET_FILE}")
print(f"  Max Words: {MAX_WORDS}")
print(f"  Max Length: {MAX_LEN}")
print(f"  Embedding Dim: {EMBED_DIM}")
print(f"  LSTM Units: {LSTM_UNITS}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch Size: {BATCH_SIZE}")
print()

# =============================================
# 1. CLEAN URL FUNCTION
# =============================================
def clean_url(url):
    """
    Clean and normalize URL for processing
    Removes protocol, www, trailing slashes, and special characters
    """
    url = str(url).lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www\.", "", url)
    url = url.rstrip("/")
    url = re.sub(r"[^a-zA-Z0-9]", " ", url)
    return url

# =============================================
# 2. LOAD AND EXPLORE DATA
# =============================================
print("="*70)
print("STEP 1: LOADING DATA")
print("="*70)

try:
    df = pd.read_csv(DATASET_FILE)
    print(f"✓ Loaded dataset: {len(df)} rows")
except FileNotFoundError:
    print(f"✗ Error: Could not find '{DATASET_FILE}'")
    print("\nPlease make sure your dataset file exists in the current directory.")
    print("Expected columns: 'url' and 'type'")
    exit(1)

# Check required columns
if 'url' not in df.columns or 'type' not in df.columns:
    print("✗ Error: Dataset must have 'url' and 'type' columns")
    print(f"Found columns: {df.columns.tolist()}")
    exit(1)

# Display dataset info
print(f"\nDataset Info:")
print(f"  Total URLs: {len(df):,}")
print(f"  Columns: {df.columns.tolist()}")

print(f"\nCategory Distribution:")
print(df['type'].value_counts())

print(f"\nCategory Percentages:")
print(df['type'].value_counts(normalize=True) * 100)

# Check for missing values
missing = df.isnull().sum()
if missing.any():
    print(f"\nMissing Values:")
    print(missing[missing > 0])
    
    # Remove rows with missing values
    df = df.dropna()
    print(f"✓ Removed rows with missing values. New size: {len(df)}")

# Display sample URLs
print(f"\nSample URLs from each category:")
for category in df['type'].unique():
    sample = df[df['type'] == category].iloc[0]['url']
    print(f"  {category:12s}: {sample[:60]}...")

# =============================================
# 3. PREPROCESS URLs
# =============================================
print("\n" + "="*70)
print("STEP 2: PREPROCESSING URLs")
print("="*70)

print("\nCleaning URLs...")
df["url"] = df["url"].astype(str).apply(clean_url)

texts = df["url"].values
labels = df["type"].values

print(f"✓ Cleaned {len(texts)} URLs")

# Show example of cleaned URL
print(f"\nExample cleaned URL:")
print(f"  Original category: {labels[0]}")
print(f"  Cleaned text: {texts[0][:80]}...")

# =============================================
# 4. ENCODE LABELS
# =============================================
print("\n" + "="*70)
print("STEP 3: ENCODING LABELS")
print("="*70)

le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
num_classes = len(le.classes_)

print(f"\nLabel Encoding:")
for i, label in enumerate(le.classes_):
    count = np.sum(labels_encoded == i)
    print(f"  {label:12s} -> {i} ({count:,} samples)")

print(f"\nTotal classes: {num_classes}")

# Save label classes
np.save(LABEL_CLASSES_FILE, le.classes_)
print(f"✓ Saved label classes to '{LABEL_CLASSES_FILE}'")

# =============================================
# 5. TOKENIZATION
# =============================================
print("\n" + "="*70)
print("STEP 4: TOKENIZATION")
print("="*70)

print(f"\nTokenizing URLs...")
print(f"  Vocabulary size: {MAX_WORDS}")
print(f"  Max sequence length: {MAX_LEN}")

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=MAX_LEN, padding="post")

print(f"\n✓ Tokenization complete:")
print(f"  Vocabulary size: {len(tokenizer.word_index)}")
print(f"  Sequences shape: {padded.shape}")

# Save tokenizer
with open(TOKENIZER_FILE, "w") as f:
    f.write(tokenizer.to_json())
print(f"✓ Saved tokenizer to '{TOKENIZER_FILE}'")

# Display tokenization example
print(f"\nExample tokenization:")
sample_idx = 0
print(f"  Original: {df.iloc[sample_idx]['url'][:60]}...")
print(f"  Cleaned: {texts[sample_idx][:60]}...")
print(f"  Tokens: {sequences[sample_idx][:20]}...")
print(f"  Padded shape: {padded[sample_idx].shape}")

# =============================================
# 6. TRAIN-TEST SPLIT
# =============================================
print("\n" + "="*70)
print("STEP 5: TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    padded, labels_encoded, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=labels_encoded
)

print(f"\n✓ Data split:")
print(f"  Training samples: {len(X_train):,}")
print(f"  Testing samples: {len(X_test):,}")
print(f"  Split ratio: {(1-TEST_SIZE)*100:.0f}% train / {TEST_SIZE*100:.0f}% test")

print(f"\nTraining set distribution:")
train_labels = pd.Series(y_train).map(dict(enumerate(le.classes_)))
print(train_labels.value_counts())

print(f"\nTest set distribution:")
test_labels = pd.Series(y_test).map(dict(enumerate(le.classes_)))
print(test_labels.value_counts())

# =============================================
# 7. COMPUTE CLASS WEIGHTS
# =============================================
print("\n" + "="*70)
print("STEP 6: COMPUTING CLASS WEIGHTS")
print("="*70)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

print(f"\nClass weights (for handling imbalance):")
for class_id, weight in class_weights_dict.items():
    class_name = le.classes_[class_id]
    print(f"  {class_name:12s}: {weight:.4f}")

# =============================================
# 8. BUILD ATTENTION LAYER
# =============================================
class AttentionLayer(Layer):
    """
    Custom Attention Layer for LSTM
    Helps the model focus on important parts of the URL
    """
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
        # Attention mechanism
        e = K.tanh(K.dot(x, self.W) + self.b)  # Attention scores
        a = K.softmax(e, axis=1)                # Attention weights
        out = x * a                             # Apply attention
        return K.sum(out, axis=1)               # Weighted sum
    
    def get_config(self):
        return super().get_config()

# =============================================
# 9. BUILD MODEL
# =============================================
print("\n" + "="*70)
print("STEP 7: BUILDING LSTM + ATTENTION MODEL")
print("="*70)

print(f"\nModel Architecture:")
print(f"  1. Input Layer: ({MAX_LEN},)")
print(f"  2. Embedding Layer: {MAX_WORDS} -> {EMBED_DIM}")
print(f"  3. LSTM Layer: {LSTM_UNITS} units (with return_sequences=True)")
print(f"  4. Attention Layer: Custom attention mechanism")
print(f"  5. Dropout Layer: {DROPOUT_RATE}")
print(f"  6. Dense Layer: 64 units (ReLU)")
print(f"  7. Output Layer: {num_classes} units (Softmax)")

# Build model
inputs = Input(shape=(MAX_LEN,), name='input')
embedding = Embedding(MAX_WORDS, EMBED_DIM, name='embedding')(inputs)
lstm_out = LSTM(LSTM_UNITS, return_sequences=True, name='lstm')(embedding)
attention_out = AttentionLayer(name='attention')(lstm_out)
dropout = Dropout(DROPOUT_RATE, name='dropout')(attention_out)
dense = Dense(64, activation="relu", name='dense')(dropout)
outputs = Dense(num_classes, activation="softmax", name='output')(dense)

model = Model(inputs, outputs, name='LSTM_Attention_Phishing_Detector')

# Compile model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
    metrics=["accuracy"]
)

print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)
model.summary()

# Count parameters
trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
print(f"\nTotal trainable parameters: {trainable_params:,}")

# =============================================
# 10. SETUP CALLBACKS
# =============================================
print("\n" + "="*70)
print("STEP 8: SETTING UP CALLBACKS")
print("="*70)

callbacks = [
    # Save best model
    ModelCheckpoint(
        BEST_MODEL,
        monitor="val_loss",
        save_best_only=True,
        mode='min',
        verbose=1
    ),
    
    # Early stopping
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Reduce learning rate on plateau
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    
    # CSV logger
    CSVLogger(TRAINING_LOG)
]

print("\nCallbacks configured:")
print(f"  ✓ ModelCheckpoint: Saves best model to '{BEST_MODEL}'")
print(f"  ✓ EarlyStopping: Stops if no improvement for 5 epochs")
print(f"  ✓ ReduceLROnPlateau: Reduces LR by 0.5 if plateau detected")
print(f"  ✓ CSVLogger: Logs training history to '{TRAINING_LOG}'")

# =============================================
# 11. TRAIN MODEL
# =============================================
print("\n" + "="*70)
print("STEP 9: TRAINING MODEL")
print("="*70)

print(f"\nStarting training with:")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training samples: {len(X_train):,}")
print(f"  Validation samples: {len(X_test):,}")
print(f"  Steps per epoch: {len(X_train) // BATCH_SIZE}")

print("\n" + "-"*70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "-"*70)
print("✓ Training complete!")

# =============================================
# 12. EVALUATE MODEL
# =============================================
print("\n" + "="*70)
print("STEP 10: EVALUATING MODEL")
print("="*70)

# Make predictions
print("\nGenerating predictions on test set...")
pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(pred_probs, axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Per-class accuracy
print("\n" + "="*70)
print("PER-CLASS METRICS")
print("="*70)
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, labels=range(num_classes)
)

print(f"\n{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
print("-"*60)
for i, class_name in enumerate(le.classes_):
    print(f"{class_name:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} "
          f"{f1[i]:<10.4f} {support[i]:<10}")

# =============================================
# 13. CONFUSION MATRIX
# =============================================
print("\n" + "="*70)
print("STEP 11: GENERATING VISUALIZATIONS")
print("="*70)

cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - LSTM + Attention Phishing Detector")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
print("✓ Saved confusion matrix to 'confusion_matrix.png'")
plt.close()

# =============================================
# 14. TRAINING CURVES
# =============================================
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", marker='o')
plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker='s')
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", marker='o')
plt.plot(history.history["val_loss"], label="Validation Loss", marker='s')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_history.png", dpi=300, bbox_inches='tight')
print("✓ Saved training history to 'training_history.png'")
plt.close()

# =============================================
# 15. SAVE MODEL AND METADATA
# =============================================
print("\n" + "="*70)
print("STEP 12: SAVING MODEL AND METADATA")
print("="*70)

# Save final model
model.save(FINAL_MODEL)
print(f"✓ Saved final model to '{FINAL_MODEL}'")

# Save metadata
metadata = {
    'model_name': MODEL_NAME,
    'dataset_file': DATASET_FILE,
    'total_samples': len(df),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'num_classes': num_classes,
    'classes': le.classes_.tolist(),
    'max_words': MAX_WORDS,
    'max_len': MAX_LEN,
    'embed_dim': EMBED_DIM,
    'lstm_units': LSTM_UNITS,
    'accuracy': float(accuracy),
    'epochs_trained': len(history.history['accuracy']),
    'best_val_accuracy': float(max(history.history['val_accuracy'])),
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open(METADATA_FILE, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Saved metadata to '{METADATA_FILE}'")

# =============================================
# 16. SAMPLE PREDICTIONS
# =============================================
print("\n" + "="*70)
print("STEP 13: SAMPLE PREDICTIONS")
print("="*70)

num_samples = min(10, len(X_test))
sample_indices = np.random.choice(len(X_test), num_samples, replace=False)

print(f"\nShowing {num_samples} random predictions:\n")

for i, idx in enumerate(sample_indices, 1):
    # Get original URL
    original_url = df.iloc[idx]['url']
    
    # Get predictions
    pred_prob = pred_probs[idx]
    pred_class_idx = y_pred[idx]
    true_class_idx = y_test[idx]
    
    pred_class = le.classes_[pred_class_idx]
    true_class = le.classes_[true_class_idx]
    confidence = pred_prob[pred_class_idx]
    
    # Display
    print(f"{i}. URL: {original_url[:60]}...")
    print(f"   True: {true_class:12s} | Predicted: {pred_class:12s} ({confidence:.2%})")
    
    if pred_class == true_class:
        print(f"   ✓ CORRECT")
    else:
        print(f"   ✗ INCORRECT")
    
    # Show top 3 predictions
    top_3_indices = np.argsort(pred_prob)[-3:][::-1]
    print(f"   Top 3 predictions:")
    for rank, idx in enumerate(top_3_indices, 1):
        class_name = le.classes_[idx]
        prob = pred_prob[idx]
        print(f"      {rank}. {class_name:12s}: {prob:.2%}")
    print()

# =============================================
# 17. SUMMARY
# =============================================
print("="*70)
print("TRAINING COMPLETE!")
print("="*70)

print(f"\n📁 Generated Files:")
print(f"  ✓ {BEST_MODEL} - Best model (based on validation loss)")
print(f"  ✓ {FINAL_MODEL} - Final model")
print(f"  ✓ {TOKENIZER_FILE} - Tokenizer for URL preprocessing")
print(f"  ✓ {LABEL_CLASSES_FILE} - Label encoder classes")
print(f"  ✓ {METADATA_FILE} - Model metadata and stats")
print(f"  ✓ {TRAINING_LOG} - Training history CSV")
print(f"  ✓ confusion_matrix.png - Confusion matrix visualization")
print(f"  ✓ training_history.png - Training curves")

print(f"\n📊 Final Model Performance:")
print(f"  Overall Accuracy: {accuracy*100:.2f}%")
print(f"  Best Val Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
print(f"  Total Parameters: {trainable_params:,}")
print(f"  Training Time: {len(history.history['accuracy'])} epochs")

print(f"\n🎯 Model Categories:")
for i, class_name in enumerate(le.classes_):
    print(f"  {i}: {class_name}")

print(f"\n🚀 Next Steps:")
print(f"  1. Use '{FINAL_MODEL}' for predictions")
print(f"  2. Run predict_new_urls.py to classify new URLs")
print(f"  3. Review confusion_matrix.png for detailed analysis")
print(f"  4. Check training_history.png for training progress")

print("\n" + "="*70)
print("🎉 YOUR PHISHING DETECTOR IS READY!")
print("="*70)