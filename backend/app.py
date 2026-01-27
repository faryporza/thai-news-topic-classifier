#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai News Topic Classifier - Backend API
=========================================
Flask API สำหรับจำแนกหมวดหมู่ข่าวภาษาไทย

Endpoints:
- GET /health - ตรวจสอบสถานะ API
- GET /model/info - ข้อมูลโมเดล
- POST /predict - ทำนายหมวดหมู่ข่าว
"""

import os
import re
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
# Load Model
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

# Load vectorizer and model
vectorizer = None
model = None
model_info = {
    "name": "Thai News Topic Classifier",
    "version": "1.0.0",
    "algorithm": "TF-IDF + Logistic Regression",
    "classes": ["Business", "SciTech", "World"],
    "created_at": "2026-01-27",
    "description": "โมเดลจำแนกหมวดหมู่ข่าวภาษาไทย 3 หมวด"
}

def load_models():
    """โหลด TF-IDF Vectorizer และ Model"""
    global vectorizer, model
    
    vectorizer_path = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.joblib')
    
    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        print(f"✅ Models loaded from {MODELS_DIR}")
        return True
    else:
        print(f"❌ Model files not found in {MODELS_DIR}")
        return False


# Load models at module level (for gunicorn)
load_models()


# ============================================================================
# Text Preprocessing (same as training)
# ============================================================================
def preprocess_text(text: str) -> str:
    """
    Preprocess ข้อความเหมือนกับตอน training
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
        "version": "1.0.0",
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
    model_loaded = vectorizer is not None and model is not None
    
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded
    }), 200 if model_loaded else 503


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """
    Model Info Endpoint
    แสดงข้อมูลโมเดล
    """
    if vectorizer is None or model is None:
        return jsonify({
            "error": "Model not loaded"
        }), 503
    
    # เพิ่มข้อมูล vocabulary size
    info = model_info.copy()
    info["vocabulary_size"] = len(vectorizer.vocabulary_)
    info["model_loaded"] = True
    
    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Endpoint
    ทำนายหมวดหมู่ข่าว
    
    Request Body:
    {
        "headline": "พาดหัวข่าว",
        "body": "เนื้อหาข่าว"
    }
    
    Response:
    {
        "label": "Business",
        "confidence": 0.95,
        "probabilities": {
            "Business": 0.95,
            "SciTech": 0.03,
            "World": 0.02
        }
    }
    """
    # ตรวจสอบว่าโหลดโมเดลแล้วหรือยัง
    if vectorizer is None or model is None:
        return jsonify({
            "error": "Model not loaded",
            "message": "กรุณาตรวจสอบว่าไฟล์โมเดลอยู่ใน /backend/models"
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
    
    # Transform ด้วย TF-IDF
    X = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    
    # สร้าง response
    classes = model.classes_
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    confidence = float(max(probabilities))
    
    return jsonify({
        "label": prediction,
        "confidence": confidence,
        "probabilities": prob_dict,
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
    print("=" * 60)
    
    # Load models
    if load_models():
        print(f"   Classes: {model.classes_}")
        print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    print("\n📡 Starting server...")
    print("   URL: http://localhost:5000")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
