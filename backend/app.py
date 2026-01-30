#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai News Topic Classifier - Backend API (WangchanBERTa Edition)
================================================================
Flask API สำหรับจำแนกหมวดหมู่ข่าวภาษาไทย ด้วย WangchanBERTa

Endpoints:
- GET /health - ตรวจสอบสถานะ API
- GET /model/info - ข้อมูลโมเดล
- POST /predict - ทำนายหมวดหมู่ข่าว

Model: farypor/my-thai-news-classifier (Hugging Face Hub)
- ความแม่นยำ: ~95-100%
- เข้าใจบริบทภาษาไทย
- รองรับ Mixed Signal และ Typo
"""

import os
import re
import time
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================================
# Flask App Configuration
# ============================================================================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================================================
# Load Model from Hugging Face Hub
# ============================================================================
MODEL_ID = "farypor/my-thai-news-classifier"

# Global model and tokenizer
tokenizer = None
model = None
device = None

model_info = {
    "name": "Thai News Topic Classifier",
    "version": "2.0.0",
    "algorithm": "WangchanBERTa",
    "model_type": "WangchanBERTa (Transformer)",
    "model_id": MODEL_ID,
    "base_model": "airesearch/wangchanberta-base-att-spm-uncased",
    "classes": ["Business", "SciTech", "World"],
    "created_at": "2026-01-30",
    "description": "โมเดลจำแนกหมวดหมู่ข่าวภาษาไทย 3 หมวด ด้วย WangchanBERTa",
    "advantages": [
        "เข้าใจบริบท (Contextual Understanding)",
        "รองรับ Mixed Signal",
        "ทนต่อ Typo",
        "Subword Tokenization"
    ],
    "accuracy": "100% (บน test set)",
    "max_length": 256
}


def load_models():
    """โหลด WangchanBERTa Model และ Tokenizer จาก Hugging Face Hub"""
    global tokenizer, model, device
    
    try:
        # Determine device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        print(f"🔧 Using device: {device}")
        
        # Load tokenizer and model from Hugging Face Hub
        print(f"📥 Loading tokenizer from {MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        print(f"📥 Loading WangchanBERTa model from {MODEL_ID}...")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        print(f"✅ WangchanBERTa model loaded successfully!")
        print(f"   - Model ID: {MODEL_ID}")
        print(f"   - Model type: {model.config.model_type}")
        print(f"   - Num labels: {model.config.num_labels}")
        print(f"   - Labels: {model.config.id2label}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


# Load models at module level (for gunicorn)
try:
    load_models()
except Exception as e:
    print(f"❌ Error loading models during startup: {e}")


# ============================================================================
# Text Preprocessing
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
        "version": "2.0.0",
        "model": "WangchanBERTa",
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
    model_loaded = tokenizer is not None and model is not None
    
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "model_type": "WangchanBERTa" if model_loaded else None,
        "device": str(device) if device else None
    }), 200 if model_loaded else 503


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """
    Model Info Endpoint
    แสดงข้อมูลโมเดล
    """
    if tokenizer is None or model is None:
        return jsonify({
            "error": "Model not loaded"
        }), 503
    
    # Build info response
    info = model_info.copy()
    info["vocabulary_size"] = len(tokenizer)
    info["model_loaded"] = True
    info["device"] = str(device)
    info["id2label"] = model.config.id2label
    info["num_parameters"] = sum(p.numel() for p in model.parameters())
    
    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Endpoint
    ทำนายหมวดหมู่ข่าวด้วย WangchanBERTa
    
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
        "latency_ms": 45,
        "model_version": "2.0.0",
        "model_type": "WangchanBERTa"
    }
    """
    start_time = time.time()
    
    # ตรวจสอบว่าโหลดโมเดลแล้วหรือยัง
    if tokenizer is None or model is None:
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
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict with WangchanBERTa
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]
        predicted_id = torch.argmax(probs).item()
    
    # Calculate latency
    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000, 2)
    
    # Build response
    predicted_label = model.config.id2label[predicted_id]
    confidence = float(probs[predicted_id])
    
    prob_dict = {
        model.config.id2label[i]: float(probs[i])
        for i in range(len(probs))
    }
    
    return jsonify({
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "latency_ms": latency_ms,
        "model_version": model_info["version"],
        "model_type": "WangchanBERTa",
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
    print("   Model: WangchanBERTa (v2.0.0)")
    print("=" * 60)
    
    # Load models
    if load_models():
        print(f"   Classes: {list(model.config.id2label.values())}")
        print(f"   Vocabulary size: {len(tokenizer)}")
        print(f"   Device: {device}")
    
    print("\n📡 Starting server...")
    print("   URL: http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
