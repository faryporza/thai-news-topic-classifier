#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai News Topic Classifier - Backend API (Logistic Regression Edition)
======================================================================
Flask API สำหรับจำแนกหมวดหมู่ข่าวภาษาไทย ด้วย TF-IDF + Logistic Regression

Endpoints:
- GET /health - ตรวจสอบสถานะ API
- GET /model/info - ข้อมูลโมเดล
- POST /predict - ทำนายหมวดหมู่ข่าว

Model: TF-IDF + Logistic Regression (joblib)
- ความแม่นยำ: ~95-100% (บน test set)
- Inference เร็วมาก (ไม่ต้องใช้ GPU)
- Cold start เร็ว (โมเดลเล็ก)
"""

import os
import re
import time
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# ============================================================================
# Flask App Configuration
# ============================================================================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================================================
# Load Model (Logistic Regression + TF-IDF)
# ============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Global model and vectorizer
vectorizer = None
model = None
model_loaded = False
model_load_error = None

model_info = {
    "name": "Thai News Topic Classifier",
    "version": "3.0.0",
    "algorithm": "TF-IDF + Logistic Regression",
    "model_type": "scikit-learn (joblib)",
    "base_model": "Logistic Regression with TF-IDF (unigram + bigram)",
    "classes": ["Business", "SciTech", "World"],
    "created_at": "2026-01-27",
    "description": "โมเดลจำแนกหมวดหมู่ข่าวภาษาไทย 3 หมวด ด้วย TF-IDF + Logistic Regression",
    "advantages": [
        "Inference เร็วมาก",
        "ไม่ต้องใช้ GPU",
        "Cold start เร็ว (โมเดลเล็ก)",
        "Deploy ง่าย"
    ],
    "accuracy": "~95-100% (บน test set)",
}


def load_models():
    """โหลด TF-IDF Vectorizer และ Logistic Regression Model จาก joblib"""
    global vectorizer, model, model_loaded, model_load_error

    try:
        vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
        model_path = os.path.join(MODEL_DIR, "logistic_regression_model.joblib")

        print(f"📥 Loading TF-IDF vectorizer from {vectorizer_path}...")
        vectorizer = joblib.load(vectorizer_path)

        print(f"📥 Loading Logistic Regression model from {model_path}...")
        model = joblib.load(model_path)

        model_loaded = True
        print(f"✅ Models loaded successfully!")
        print(f"   - Vectorizer vocab size: {len(vectorizer.vocabulary_):,}")
        print(f"   - Model classes: {list(model.classes_)}")
        return True

    except Exception as e:
        model_load_error = str(e)
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


# Load models at startup (fast — joblib models are small)
print("🚀 Starting server... Loading models...")
load_models()


# ============================================================================
# Text Preprocessing (same as training)
# ============================================================================
def preprocess_text(text: str) -> str:
    """
    Preprocess ข้อความ (ทำให้เหมือนกับตอน training)
    """
    if text is None:
        return ""

    text = str(text)

    # 1. Whitespace Normalization
    text = re.sub(r'\s+', ' ', text)

    # 2. Strip
    text = text.strip()

    # 3. Thai Digits Normalization
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'
    arabic_digits = '0123456789'
    for thai, arabic in zip(thai_digits, arabic_digits):
        text = text.replace(thai, arabic)

    return text


# ============================================================================
# API Endpoints
# ============================================================================
@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Thai News Topic Classifier API",
        "version": "3.0.0",
        "model": "TF-IDF + Logistic Regression",
        "endpoints": [
            "GET /health",
            "GET /model/info",
            "POST /predict"
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    """
    Health Check Endpoint
    ตรวจสอบสถานะ API และโมเดล
    """
    if model_loaded:
        status = "healthy"
    elif model_load_error:
        status = "error"
    else:
        status = "starting"

    return jsonify({
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "model_type": "TF-IDF + Logistic Regression" if model_loaded else None,
        "error": model_load_error
    }), 200  # Always return 200 so startup probe passes


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """
    Model Info Endpoint
    แสดงข้อมูลโมเดล
    """
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded"
        }), 503

    info = model_info.copy()
    info["vocabulary_size"] = len(vectorizer.vocabulary_)
    info["model_loaded"] = True
    info["model_classes"] = list(model.classes_)

    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Endpoint
    ทำนายหมวดหมู่ข่าวด้วย TF-IDF + Logistic Regression

    Request Body:
    {
        "headline": "พาดหัวข่าว",
        "body": "เนื้อหาข่าว"
    }

    Response:
    {
        "label": "Business",
        "confidence": 0.95,
        "probabilities": {...},
        "latency_ms": 5,
        "model_version": "3.0.0",
        "model_type": "TF-IDF + Logistic Regression"
    }
    """
    start_time = time.time()

    # ตรวจสอบว่าโหลดโมเดลแล้วหรือยัง
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded",
            "message": "กรุณารอโมเดลโหลดเสร็จ"
        }), 503

    # รับ request data
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No data provided",
            "message": "กรุณาส่ง JSON body"
        }), 400

    headline = data.get('headline', '')
    body = data.get('body', '')

    if not headline and not body:
        return jsonify({
            "error": "Missing required fields",
            "message": "กรุณากรอก headline หรือ body อย่างน้อย 1 อย่าง"
        }), 400

    # รวม headline และ body
    text = headline + ' ' + body

    # Preprocess
    text = preprocess_text(text)

    # TF-IDF transform
    X = vectorizer.transform([text])

    # Predict
    predicted_label = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Calculate latency
    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000, 2)

    # Build probability dict
    prob_dict = {
        cls: float(prob)
        for cls, prob in zip(model.classes_, probabilities)
    }

    confidence = float(max(probabilities))

    return jsonify({
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "latency_ms": latency_ms,
        "model_version": model_info["version"],
        "model_type": "TF-IDF + Logistic Regression",
        "input": {
            "headline": headline[:100] + "..." if len(headline) > 100 else headline,
            "body": body[:200] + "..." if len(body) > 200 else body
        }
    })


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🇹🇭 Thai News Topic Classifier API")
    print("   Model: TF-IDF + Logistic Regression (v3.0.0)")
    print("=" * 60)

    print("\n📡 Starting server...")
    print("   URL: http://localhost:5001")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5001, debug=True)