"""
PHISHGUARD AI - FASTAPI BACKEND (FIXED)
========================================

FastAPI backend for LSTM + Attention Phishing Detection Model
Fixed model loading with proper error handling

Author: PhishGuard Team
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, validator
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import json
import re
import io
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
# ATTENTION LAYER DEFINITION
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
def clean_url(url: str) -> str:
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
    Singleton phishing detection model handler
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PhishingDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.tokenizer = None
        self.label_classes = None
        self.metadata = None
        self.max_len = None
        
        self._load_components()
        self._initialized = True
    
    def _load_components(self):
        """
        Load model, tokenizer, and metadata with improved error handling
        """
        print("="*70)
        print("LOADING PHISHING DETECTOR")
        print("="*70)
        
        # Load metadata
        try:
            with open(METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)
            self.max_len = self.metadata['max_len']
            print(f"✓ Loaded metadata from '{METADATA_FILE}'")
            print(f"  Model: {self.metadata['model_name']}")
            print(f"  Accuracy: {self.metadata['accuracy']*100:.2f}%")
        except FileNotFoundError:
            print(f"⚠ Warning: Metadata file '{METADATA_FILE}' not found")
            print("  Using default max_len=120")
            self.max_len = 120
        except Exception as e:
            print(f"⚠ Warning: Error loading metadata: {e}")
            self.max_len = 120
        
        # Load label classes
        try:
            self.label_classes = np.load(LABEL_CLASSES_FILE, allow_pickle=True)
            print(f"✓ Loaded {len(self.label_classes)} label classes")
        except FileNotFoundError:
            print(f"✗ Error: Label classes file '{LABEL_CLASSES_FILE}' not found")
            raise
        except Exception as e:
            print(f"✗ Error loading label classes: {e}")
            raise
        
        # Load tokenizer
        try:
            with open(TOKENIZER_FILE, 'r') as f:
                tokenizer_json = f.read()
            self.tokenizer = tokenizer_from_json(tokenizer_json)
            print(f"✓ Loaded tokenizer from '{TOKENIZER_FILE}'")
        except FileNotFoundError:
            print(f"✗ Error: Tokenizer file '{TOKENIZER_FILE}' not found")
            raise
        except Exception as e:
            print(f"✗ Error loading tokenizer: {e}")
            raise
        
        # Load model with comprehensive error handling
        try:
            # First, check if model file exists
            model_path = Path(MODEL_FILE)
            if not model_path.exists():
                print(f"✗ Error: Model file '{MODEL_FILE}' not found")
                print(f"  Current directory: {Path.cwd()}")
                available_models = list(Path.cwd().glob('*.keras')) + list(Path.cwd().glob('*.h5'))
                print(f"  Available model files: {[str(m) for m in available_models]}")
                raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
            
            print(f"  Loading model from '{MODEL_FILE}'...")
            print(f"  Model file size: {model_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Load with custom objects and compile=False for inference
            self.model = load_model(
                MODEL_FILE,
                custom_objects={'AttentionLayer': AttentionLayer},
                compile=False
            )
            
            # Recompile for inference (optional but recommended)
            self.model.compile(
                loss="sparse_categorical_crossentropy",
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                metrics=["accuracy"]
            )
            
            print(f"✓ Loaded model from '{MODEL_FILE}'")
            
            # Test the model with dummy input
            print(f"  Testing model inference...")
            dummy_input = np.zeros((1, self.max_len), dtype=np.int32)
            test_pred = self.model.predict(dummy_input, verbose=0)
            print(f"✓ Model inference test successful (output shape: {test_pred.shape})")
            
        except FileNotFoundError as e:
            print(f"✗ Model file not found: {e}")
            raise
        except Exception as e:
            print(f"✗ Failed to initialize detector: {e}")
            print(f"  Error type: {type(e).__name__}")
            import traceback
            print(f"  Traceback:\n{traceback.format_exc()}")
            raise
        
        print("\n✓ All components loaded successfully!")
        print("="*70 + "\n")
    
    def preprocess_url(self, url: str) -> np.ndarray:
        """
        Preprocess a single URL for prediction
        """
        cleaned = clean_url(url)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post')
        return padded
    
    def predict_single(self, url: str) -> Dict:
        """
        Predict category for a single URL
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
            
        processed = self.preprocess_url(url)
        predictions = self.model.predict(processed, verbose=0)[0]
        
        predicted_class_idx = np.argmax(predictions)
        predicted_class = self.label_classes[predicted_class_idx]
        confidence = predictions[predicted_class_idx]
        
        result = {
            'url': url,
            'predicted_category': str(predicted_class),
            'confidence': float(confidence),
            'is_safe': str(predicted_class).lower() == 'benign',
            'all_probabilities': {
                str(self.label_classes[i]): float(predictions[i])
                for i in range(len(self.label_classes))
            }
        }
        
        return result
    
    def predict_batch(self, urls: List[str]) -> List[Dict]:
        """
        Predict categories for multiple URLs
        """
        results = []
        for url in urls:
            result = self.predict_single(url)
            results.append(result)
        return results

# =============================================
# PYDANTIC MODELS
# =============================================
class URLRequest(BaseModel):
    """Single URL prediction request"""
    url: str
    
    @validator('url')
    def validate_url(cls, v):
        if not v or not v.strip():
            raise ValueError('URL cannot be empty')
        return v.strip()

class BatchURLRequest(BaseModel):
    """Batch URL prediction request"""
    urls: List[str]
    
    @validator('urls')
    def validate_urls(cls, v):
        if not v or len(v) == 0:
            raise ValueError('URLs list cannot be empty')
        if len(v) > 1000:
            raise ValueError('Maximum 1000 URLs per batch')
        return [url.strip() for url in v if url.strip()]

class PredictionResponse(BaseModel):
    """Single prediction response"""
    url: str
    predicted_category: str
    confidence: float
    is_safe: bool
    all_probabilities: Dict[str, float]

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    results: List[PredictionResponse]
    total_urls: int
    safe_count: int
    threat_count: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
    accuracy: Optional[float] = None
    classes: Optional[List[str]] = None
    error: Optional[str] = None

# =============================================
# FASTAPI APPLICATION
# =============================================
app = FastAPI(
    title="PhishGuard AI API",
    description="LSTM + Attention based Phishing Detection API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector (singleton)
detector = None
initialization_error = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the model on startup
    """
    global detector, initialization_error
    try:
        detector = PhishingDetector()
        print("✓ API server ready!")
    except Exception as e:
        initialization_error = str(e)
        print(f"✗ Failed to initialize detector: {e}")
        print("API will start but predictions will fail")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if model file exists: {MODEL_FILE}")
        print(f"  2. Verify all required files are present:")
        print(f"     - {MODEL_FILE}")
        print(f"     - {TOKENIZER_FILE}")
        print(f"     - {LABEL_CLASSES_FILE}")
        print(f"     - {METADATA_FILE}")
        print(f"  3. Ensure model was saved using: model.save('model.keras')")

# =============================================
# FRONTEND SERVING
# =============================================
@app.get("/", response_class=HTMLResponse, tags=["General"])
async def serve_frontend():
    """
    Serve the frontend HTML page
    """
    html_file = Path("index.html")
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content=f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PhishGuard AI</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .error {{ background: #ffebee; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .info {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>🛡️ PhishGuard AI API</h1>
            {"<div class='error'><strong>⚠️ Model Error:</strong><br>" + initialization_error + "</div>" if initialization_error else ""}
            <div class='info'>
                <p><strong>API Documentation:</strong> <a href="/docs">/docs</a></p>
                <p><strong>Health Check:</strong> <a href="/health">/health</a></p>
            </div>
            <p>Frontend HTML not found. Please ensure index.html is in the same directory.</p>
        </body>
        </html>
        """)

# =============================================
# API ENDPOINTS
# =============================================

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    """
    if detector is None or detector.model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            error=initialization_error
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=detector.metadata.get('model_name') if detector.metadata else None,
        accuracy=detector.metadata.get('accuracy') if detector.metadata else None,
        classes=detector.label_classes.tolist() if detector.label_classes is not None else None
    )

@app.post("/api/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_url(request: URLRequest):
    """
    Predict category for a single URL
    
    - **url**: The URL to analyze
    
    Returns prediction with confidence scores for all categories
    """
    if detector is None or detector.model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded. Error: {initialization_error}"
        )
    
    try:
        result = detector.predict_single(request.url)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchURLRequest):
    """
    Predict categories for multiple URLs
    
    - **urls**: List of URLs to analyze (max 1000)
    
    Returns predictions for all URLs with summary statistics
    """
    if detector is None or detector.model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded. Error: {initialization_error}"
        )
    
    try:
        results = detector.predict_batch(request.urls)
        
        # Calculate statistics
        safe_count = sum(1 for r in results if r['is_safe'])
        threat_count = len(results) - safe_count
        
        return BatchPredictionResponse(
            results=[PredictionResponse(**r) for r in results],
            total_urls=len(results),
            safe_count=safe_count,
            threat_count=threat_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/api/predict/file", tags=["Prediction"])
async def predict_from_file(file: UploadFile = File(...)):
    """
    Predict categories for URLs from uploaded file
    
    Supports:
    - Text files (.txt) - one URL per line
    - CSV files (.csv) - must have 'url' column
    
    Returns predictions in JSON format
    """
    if detector is None or detector.model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded. Error: {initialization_error}"
        )
    
    try:
        content = await file.read()
        
        # Handle text file
        if file.filename.endswith('.txt'):
            text = content.decode('utf-8')
            urls = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Handle CSV file
        elif file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            if 'url' not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"CSV must have 'url' column. Found: {df.columns.tolist()}"
                )
            urls = df['url'].tolist()
        
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Use .txt or .csv"
            )
        
        if len(urls) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Maximum 1000 URLs per file"
            )
        
        # Predict
        results = detector.predict_batch(urls)
        
        # Calculate statistics
        safe_count = sum(1 for r in results if r['is_safe'])
        threat_count = len(results) - safe_count
        
        return {
            "filename": file.filename,
            "total_urls": len(results),
            "safe_count": safe_count,
            "threat_count": threat_count,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.get("/api/stats", tags=["Statistics"])
async def get_model_stats():
    """
    Get model statistics and information
    """
    if detector is None or detector.model is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded. Error: {initialization_error}"
        )
    
    stats = {
        "model_info": {
            "name": detector.metadata.get('model_name') if detector.metadata else "LSTM + Attention",
            "accuracy": detector.metadata.get('accuracy') if detector.metadata else None,
            "max_sequence_length": detector.max_len,
        },
        "categories": detector.label_classes.tolist() if detector.label_classes is not None else [],
        "preprocessing": {
            "method": "URL tokenization with padding",
            "max_length": detector.max_len
        }
    }
    
    return stats

@app.get("/api/categories", tags=["Information"])
async def get_categories():
    """
    Get list of all detection categories
    """
    if detector is None or detector.label_classes is None:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded. Error: {initialization_error}"
        )
    
    return {
        "categories": detector.label_classes.tolist(),
        "count": len(detector.label_classes)
    }

# =============================================
# ERROR HANDLERS
# =============================================
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )