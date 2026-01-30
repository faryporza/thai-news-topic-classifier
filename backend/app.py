#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai News Topic Classifier - Backend API (ONNX Edition)
=======================================================
Flask API สำหรับจำแนกหมวดหมู่ข่าวภาษาไทย ด้วย ONNX Runtime

Endpoints:
- GET /health - ตรวจสอบสถานะ API
- GET /model/info - ข้อมูลโมเดล
- POST /predict - ทำนายหมวดหมู่ข่าว

Model: farypor/my-thai-news-classifier-onnx (ONNX on Hugging Face Hub)
- ความแม่นยำ: ~95-100%
- Inference เร็วกว่า PyTorch 2-3x
- ไม่ต้องใช้ GPU
"""

import os
import re
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from huggingface_hub import snapshot_download
import onnxruntime as ort
from transformers import AutoTokenizer

# ============================================================================
# Flask App Configuration
# ============================================================================
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ============================================================================
# Load Model from Hugging Face Hub (ONNX)
# ============================================================================
REPO_ID = "farypor/my-thai-news-classifier-onnx"
LOCAL_DIR = "onnx_model"
MAX_LENGTH = 256

# Global model and tokenizer
tokenizer = None
session = None

# Label mapping
ID2LABEL = {0: "Business", 1: "SciTech", 2: "World"}
LABEL2ID = {"Business": 0, "SciTech": 1, "World": 2}

model_info = {
    "name": "Thai News Topic Classifier",
    "version": "2.1.0",
    "algorithm": "WangchanBERTa + ONNX",
    "model_type": "ONNX Runtime",
    "model_id": REPO_ID,
    "base_model": "airesearch/wangchanberta-base-att-spm-uncased",
    "classes": ["Business", "SciTech", "World"],
    "created_at": "2026-01-31",
    "description": "โมเดลจำแนกหมวดหมู่ข่าวภาษาไทย 3 หมวด ด้วย ONNX Runtime",
    "advantages": [
        "Inference เร็วกว่า PyTorch 2-3x",
        "ไม่ต้องใช้ GPU",
        "ไฟล์เล็กลง",
        "Deploy ง่าย"
    ],
    "accuracy": "100% (บน test set)",
    "max_length": MAX_LENGTH
}


def load_models():
    """โหลด ONNX Model และ Tokenizer จาก Hugging Face Hub"""
    global tokenizer, session
    
    try:
        # Download model from Hugging Face Hub
        print(f"📥 Downloading model from {REPO_ID}...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"   ✅ Downloaded to {LOCAL_DIR}")
        
        # Load tokenizer
        tokenizer_path = os.path.join(LOCAL_DIR, "tokenizer")
        print(f"📥 Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        
        # Load ONNX model
        onnx_path = os.path.join(LOCAL_DIR, "model.onnx")
        print(f"📥 Loading ONNX model from {onnx_path}...")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        
        print(f"✅ ONNX model loaded successfully!")
        print(f"   - Model ID: {REPO_ID}")
        print(f"   - Provider: CPUExecutionProvider")
        print(f"   - Labels: {ID2LABEL}")
        
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


def softmax(x):
    """Compute softmax values"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ============================================================================
# API Endpoints
# ============================================================================
@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Thai News Topic Classifier API",
        "version": "2.1.0",
        "model": "ONNX Runtime",
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
    model_loaded = tokenizer is not None and session is not None
    
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "model_type": "ONNX Runtime" if model_loaded else None,
        "provider": "CPUExecutionProvider" if model_loaded else None
    }), 200 if model_loaded else 503


@app.route('/model/info', methods=['GET'])
def get_model_info():
    """
    Model Info Endpoint
    แสดงข้อมูลโมเดล
    """
    if tokenizer is None or session is None:
        return jsonify({
            "error": "Model not loaded"
        }), 503
    
    # Build info response
    info = model_info.copy()
    info["vocabulary_size"] = len(tokenizer)
    info["model_loaded"] = True
    info["id2label"] = ID2LABEL
    
    return jsonify(info)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict Endpoint
    ทำนายหมวดหมู่ข่าวด้วย ONNX Runtime
    
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
        "latency_ms": 15,
        "model_version": "2.1.0",
        "model_type": "ONNX Runtime"
    }
    """
    start_time = time.time()
    
    # ตรวจสอบว่าโหลดโมเดลแล้วหรือยัง
    if tokenizer is None or session is None:
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
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    
    # Run ONNX inference
    logits = session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
    )[0]
    
    # Calculate probabilities
    probs = softmax(logits)[0]
    predicted_id = int(np.argmax(probs))
    
    # Calculate latency
    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000, 2)
    
    # Build response
    predicted_label = ID2LABEL[predicted_id]
    confidence = float(probs[predicted_id])
    
    prob_dict = {
        ID2LABEL[i]: float(probs[i])
        for i in range(len(probs))
    }
    
    return jsonify({
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "latency_ms": latency_ms,
        "model_version": model_info["version"],
        "model_type": "ONNX Runtime",
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
    print("   Model: ONNX Runtime (v2.1.0)")
    print("=" * 60)
    
    # Load models
    if load_models():
        print(f"   Classes: {list(ID2LABEL.values())}")
        print(f"   Vocabulary size: {len(tokenizer)}")
    
    print("\n📡 Starting server...")
    print("   URL: http://localhost:5001")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
