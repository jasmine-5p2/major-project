"""
LSTM + ATTENTION PHISHING DETECTION - PREDICTION SCRIPT
========================================================

Classify new URLs using the trained LSTM + Attention model.
Supports single URL, batch URLs, and CSV file predictions.

Usage:
    python prediction.py                    # Interactive mode
    python prediction.py --url "http://example.com"
    python prediction.py --file urls.txt
    python prediction.py --csv input.csv --output results.csv
"""

import numpy as np
import pandas as pd
import json
import re
import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

# =============================================
# CONFIGURATION
# =============================================
MODEL_FILE = "final_lstm_attention_phishing_detector.keras"
TOKENIZER_FILE = "tokenizer.json"
LABEL_CLASSES_FILE = "label_classes.npy"
METADATA_FILE = "model_metadata.json"

# =============================================
# ATTENTION LAYER (Must be defined for model loading)
# =============================================
class AttentionLayer(Layer):
    """
    Custom Attention Layer - Required for loading the trained model
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
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        out = x * a
        return K.sum(out, axis=1)
    
    def get_config(self):
        return super().get_config()

# =============================================
# URL CLEANING FUNCTION
# =============================================
def clean_url(url):
    """
    Clean and normalize URL for processing
    Same preprocessing as used in training
    """
    url = str(url).lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www\.", "", url)
    url = url.rstrip("/")
    url = re.sub(r"[^a-zA-Z0-9]", " ", url)
    return url

# =============================================
# PHISHING DETECTOR CLASS
# =============================================
class PhishingDetector:
    """
    Complete phishing detection system
    Loads model, tokenizer, and handles predictions
    """
    
    def __init__(self, model_path=MODEL_FILE, tokenizer_path=TOKENIZER_FILE,
                 label_classes_path=LABEL_CLASSES_FILE, metadata_path=METADATA_FILE):
        """
        Initialize the detector by loading all necessary files
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_classes_path = label_classes_path
        self.metadata_path = metadata_path
        
        self.model = None
        self.tokenizer = None
        self.label_classes = None
        self.metadata = None
        self.max_len = None
        
        self._load_components()
    
    def _load_components(self):
        """
        Load model, tokenizer, and metadata
        """
        print("="*70)
        print("LOADING PHISHING DETECTOR")
        print("="*70)
        
        # Load metadata
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.max_len = self.metadata['max_len']
            print(f"✓ Loaded metadata from '{self.metadata_path}'")
            print(f"  Model: {self.metadata['model_name']}")
            print(f"  Accuracy: {self.metadata['accuracy']*100:.2f}%")
            print(f"  Classes: {', '.join(self.metadata['classes'])}")
        except FileNotFoundError:
            print(f"✗ Warning: Metadata file '{self.metadata_path}' not found")
            print("  Using default max_len=120")
            self.max_len = 120
        
        # Load label classes
        try:
            self.label_classes = np.load(self.label_classes_path, allow_pickle=True)
            print(f"✓ Loaded {len(self.label_classes)} label classes from '{self.label_classes_path}'")
        except FileNotFoundError:
            print(f"✗ Error: Label classes file '{self.label_classes_path}' not found")
            raise
        
        # Load tokenizer
        try:
            with open(self.tokenizer_path, 'r') as f:
                tokenizer_json = f.read()
            self.tokenizer = tokenizer_from_json(tokenizer_json)
            print(f"✓ Loaded tokenizer from '{self.tokenizer_path}'")
        except FileNotFoundError:
            print(f"✗ Error: Tokenizer file '{self.tokenizer_path}' not found")
            raise
        
        # Load model
        try:
            self.model = load_model(
                self.model_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            print(f"✓ Loaded model from '{self.model_path}'")
        except FileNotFoundError:
            print(f"✗ Error: Model file '{self.model_path}' not found")
            raise
        
        print("\n✓ All components loaded successfully!")
        print("="*70 + "\n")
    
    def preprocess_url(self, url):
        """
        Preprocess a single URL for prediction
        """
        # Clean URL
        cleaned = clean_url(url)
        
        # Tokenize
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        
        # Pad
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')
        
        return padded
    
    def predict_single(self, url, show_probabilities=True):
        """
        Predict category for a single URL
        
        Returns:
            dict with prediction results
        """
        # Preprocess
        processed = self.preprocess_url(url)
        
        # Predict
        predictions = self.model.predict(processed, verbose=0)[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.label_classes[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        # Build result
        result = {
            'url': url,
            'predicted_category': predicted_class,
            'confidence': float(confidence),
            'is_safe': predicted_class.lower() == 'benign'
        }
        
        if show_probabilities:
            result['all_probabilities'] = {
                self.label_classes[i]: float(predictions[i])
                for i in range(len(self.label_classes))
            }
        
        return result
    
    def predict_batch(self, urls, show_progress=True):
        """
        Predict categories for multiple URLs
        
        Returns:
            list of prediction dictionaries
        """
        results = []
        
        if show_progress:
            print(f"Processing {len(urls)} URLs...")
        
        for i, url in enumerate(urls, 1):
            if show_progress and i % 100 == 0:
                print(f"  Processed {i}/{len(urls)} URLs...")
            
            result = self.predict_single(url, show_probabilities=True)
            results.append(result)
        
        if show_progress:
            print(f"✓ Completed processing {len(urls)} URLs\n")
        
        return results
    
    def predict_from_file(self, filepath, output_file=None):
        """
        Predict categories for URLs from a text file (one URL per line)
        """
        print(f"Reading URLs from '{filepath}'...")
        
        # Read URLs
        with open(filepath, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"✓ Loaded {len(urls)} URLs\n")
        
        # Predict
        results = self.predict_batch(urls)
        
        # Save if output file specified
        if output_file:
            self.save_results(results, output_file)
        
        return results
    
    def predict_from_csv(self, csv_path, url_column='url', output_path=None):
        """
        Predict categories for URLs from a CSV file
        """
        print(f"Reading CSV from '{csv_path}'...")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        if url_column not in df.columns:
            raise ValueError(f"Column '{url_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        urls = df[url_column].tolist()
        print(f"✓ Loaded {len(urls)} URLs from column '{url_column}'\n")
        
        # Predict
        results = self.predict_batch(urls)
        
        # Add predictions to dataframe
        df['predicted_category'] = [r['predicted_category'] for r in results]
        df['confidence'] = [r['confidence'] for r in results]
        df['is_safe'] = [r['is_safe'] for r in results]
        
        # Add probability columns for each class
        for class_name in self.label_classes:
            df[f'prob_{class_name}'] = [
                r['all_probabilities'][class_name] for r in results
            ]
        
        # Save if output path specified
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✓ Saved results to '{output_path}'")
        
        return df
    
    def save_results(self, results, output_file):
        """
        Save prediction results to a file
        """
        if output_file.endswith('.csv'):
            # Save as CSV
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
        elif output_file.endswith('.json'):
            # Save as JSON
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            # Save as text
            with open(output_file, 'w') as f:
                for result in results:
                    f.write(f"{result['url']}\t{result['predicted_category']}\t{result['confidence']:.4f}\n")
        
        print(f"✓ Saved {len(results)} results to '{output_file}'")
    
    def display_prediction(self, result):
        """
        Display a single prediction in a formatted way
        """
        print("─" * 70)
        print(f"URL: {result['url']}")
        print(f"Predicted Category: {result['predicted_category'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Status: {'✓ SAFE' if result['is_safe'] else '⚠ POTENTIALLY DANGEROUS'}")
        
        if 'all_probabilities' in result:
            print("\nAll Probabilities:")
            for category, prob in sorted(result['all_probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True):
                bar_length = int(prob * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f"  {category:12s}: {bar} {prob:.2%}")
        print("─" * 70)

# =============================================
# INTERACTIVE MODE
# =============================================
def interactive_mode(detector):
    """
    Interactive mode for classifying URLs
    """
    print("\n" + "="*70)
    print("INTERACTIVE PHISHING DETECTION MODE")
    print("="*70)
    print("\nEnter URLs to classify (type 'quit' or 'exit' to stop)")
    print("Type 'batch' to enter multiple URLs")
    print("Type 'help' for more options\n")
    
    while True:
        try:
            user_input = input("\n🔍 Enter URL: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Enter a URL to classify it")
                print("  - 'batch' - Enter multiple URLs (one per line, empty line to finish)")
                print("  - 'quit' or 'exit' - Exit the program")
                continue
            
            if user_input.lower() == 'batch':
                print("\n📝 Enter URLs (one per line, empty line when done):")
                urls = []
                while True:
                    url = input().strip()
                    if not url:
                        break
                    urls.append(url)
                
                if urls:
                    print(f"\nProcessing {len(urls)} URLs...\n")
                    results = detector.predict_batch(urls, show_progress=False)
                    
                    for result in results:
                        detector.display_prediction(result)
                    
                    # Summary
                    print("\n" + "="*70)
                    print("BATCH SUMMARY")
                    print("="*70)
                    categories = {}
                    for result in results:
                        cat = result['predicted_category']
                        categories[cat] = categories.get(cat, 0) + 1
                    
                    for cat, count in sorted(categories.items()):
                        print(f"  {cat:12s}: {count} URLs")
                    print("="*70)
                continue
            
            # Single URL prediction
            result = detector.predict_single(user_input)
            print()
            detector.display_prediction(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")

# =============================================
# COMMAND LINE INTERFACE
# =============================================
def main():
    """
    Main function with command line argument parsing
    """
    parser = argparse.ArgumentParser(
        description='LSTM + Attention Phishing URL Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prediction.py
  python prediction.py --url "http://suspicious-site.com/login"
  python prediction.py --file urls.txt --output results.json
  python prediction.py --csv input.csv --output predictions.csv
        """
    )
    
    parser.add_argument('--url', type=str, help='Single URL to classify')
    parser.add_argument('--file', type=str, help='Text file with URLs (one per line)')
    parser.add_argument('--csv', type=str, help='CSV file with URLs')
    parser.add_argument('--url-column', type=str, default='url', 
                       help='Column name for URLs in CSV (default: url)')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--model', type=str, default=MODEL_FILE,
                       help=f'Path to model file (default: {MODEL_FILE})')
    parser.add_argument('--batch', action='store_true',
                       help='Enable batch mode for multiple URLs')
    
    args = parser.parse_args()
    
    # Initialize detector
    try:
        detector = PhishingDetector(model_path=args.model)
    except Exception as e:
        print(f"\n✗ Failed to load detector: {e}")
        print("\nMake sure the following files exist:")
        print(f"  - {MODEL_FILE}")
        print(f"  - {TOKENIZER_FILE}")
        print(f"  - {LABEL_CLASSES_FILE}")
        return
    
    # Handle different modes
    if args.url:
        # Single URL mode
        print("\nClassifying single URL...\n")
        result = detector.predict_single(args.url)
        detector.display_prediction(result)
        
        if args.output:
            detector.save_results([result], args.output)
    
    elif args.file:
        # File mode
        results = detector.predict_from_file(args.file, args.output)
        
        # Display summary
        print("\n" + "="*70)
        print("PREDICTION SUMMARY")
        print("="*70)
        categories = {}
        for result in results:
            cat = result['predicted_category']
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            percentage = (count / len(results)) * 100
            print(f"  {cat:12s}: {count:5d} URLs ({percentage:5.1f}%)")
        print("="*70)
    
    elif args.csv:
        # CSV mode
        df = detector.predict_from_csv(args.csv, args.url_column, args.output)
        
        # Display summary
        print("\n" + "="*70)
        print("PREDICTION SUMMARY")
        print("="*70)
        print(df['predicted_category'].value_counts())
        print("\nAverage confidence by category:")
        print(df.groupby('predicted_category')['confidence'].mean())
        print("="*70)
    
    else:
        # Interactive mode
        interactive_mode(detector)

# =============================================
# RUN
# =============================================
if __name__ == "__main__":
    main()