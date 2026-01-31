#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thai News Topic Classifier - Backend API (ONNX Edition)
Flask API สำหรับจำแนกหมวดหมู่ข่าวภาษาไทย ด้วย ONNX Runtime
"""

import os
import re
import time
import json
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import snapshot_download
import onnxruntime as ort
from transformers import AutoTokenizer

# ============================================================================
# Flask App Configuration
# ============================================================================
app = Flask(__name__)
CORS(app)

# ============================================================================
# Model Configuration
# ============================================================================
REPO_ID = "farypor/my-thai-news-classifier-onnx"
LOCAL_DIR = "onnx_model"
MAX_LENGTH = 192

# Debug flag (export DEBUG_PREDICT=1)
DEBUG_PREDICT = os.getenv("DEBUG_PREDICT", "0") == "1"

# Global model and tokenizer
tokenizer = None
session = None

# Default label mapping (fallback)
ID2LABEL = {0: "Business", 1: "SciTech", 2: "World"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

model_info = {
    "name": "Thai News Topic Classifier",
    "version": "2.1.1",
    "algorithm": "WangchanBERTa + ONNX",
    "model_type": "ONNX Runtime",
    "model_id": REPO_ID,
    "base_model": "airesearch/wangchanberta-base-att-spm-uncased",
    "classes": ["Business", "SciTech", "World"],
    "created_at": "2026-01-31",
    "description": "โมเดลจำแนกหมวดหมู่ข่าวภาษาไทย 3 หมวด ด้วย ONNX Runtime",
    "max_length": MAX_LENGTH
}


# ============================================================================
# Helpers
# ============================================================================
def normalize_label_name(x: str) -> str:
    """Normalize label to canonical set: Business / SciTech / World"""
    if x is None:
        return ""
    t = str(x).strip()
    low = t.lower()
    if low == "business":
        return "Business"
    if low in ("scitech", "sci-tech", "sci_tech", "science", "technology", "tech"):
        return "SciTech"
    if low == "world":
        return "World"
    # If already correct or unknown label, return original stripped
    return t


def load_label_mapping_from_config(config_path: str):
    """
    Try load id2label/label2id from Transformers config.json.
    Many exports include this; if present, it's the source of truth.
    """
    global ID2LABEL, LABEL2ID

    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        id2label_raw = cfg.get("id2label")
        label2id_raw = cfg.get("label2id")

        # Normalize shapes: keys might be string numbers in JSON
        if isinstance(id2label_raw, dict) and len(id2label_raw) > 0:
            tmp = {}
            for k, v in id2label_raw.items():
                try:
                    ik = int(k)
                except Exception:
                    continue
                tmp[ik] = normalize_label_name(v)
            if len(tmp) >= 2:
                ID2LABEL = dict(sorted(tmp.items(), key=lambda kv: kv[0]))
                LABEL2ID = {v: k for k, v in ID2LABEL.items()}
                return True

        # fallback: build id2label from label2id if present
        if isinstance(label2id_raw, dict) and len(label2id_raw) > 0:
            tmp = {}
            for lbl, idx in label2id_raw.items():
                try:
                    iidx = int(idx)
                except Exception:
                    continue
                tmp[iidx] = normalize_label_name(lbl)
            if len(tmp) >= 2:
                ID2LABEL = dict(sorted(tmp.items(), key=lambda kv: kv[0]))
                LABEL2ID = {v: k for k, v in ID2LABEL.items()}
                return True

        return False

    except Exception as e:
        print(f"⚠️  Failed to load label mapping from config: {e}")
        return False


def softmax_logits(logits_2d: np.ndarray) -> np.ndarray:
    """Stable softmax for 2D logits"""
    x = logits_2d - np.max(logits_2d, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def ensure_probs(output_2d: np.ndarray) -> np.ndarray:
    """
    Some ONNX exports might already output probabilities.
    Detect: values in [0,1] and row sum ~ 1.
    """
    out = output_2d
    if out.ndim != 2:
        out = np.array(out)
        if out.ndim == 1:
            out = out.reshape(1, -1)

    row = out[0]
    if np.all(row >= -1e-6) and np.all(row <= 1.0 + 1e-6):
        s = float(np.sum(row))
        if 0.98 <= s <= 1.02:
            return out  # already probabilities
    return softmax_logits(out)


# ============================================================================
# Text Preprocessing
# ============================================================================
def preprocess_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)

    # whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    # Thai digits -> arabic digits
    thai_digits = "๐๑๒๓๔๕๖๗๘๙"
    arabic_digits = "0123456789"
    for th, ar in zip(thai_digits, arabic_digits):
        text = text.replace(th, ar)

    return text


# ============================================================================
# Keyword-based Post Processing (Hybrid)
# ============================================================================
KEYWORDS = {
    "Business": [
        "ธนาคาร", "ธอส", "สินเชื่อ", "ดอกเบี้ย", "ค่าเงินบาท", "เงินบาท", "ดอลลาร์", "อัตราแลกเปลี่ยน",
        "หุ้น", "ตลาด", "ลงทุน", "งบ", "กำไร", "ขาดทุน", "รายได้", "เศรษฐกิจ", "พาณิชย์",
        "ส่งออก", "นำเข้า", "ราค", "ภาษี", "ผู้บริโภค", "อีคอมเมิร์ซ", "โซเชียลคอมเมิร์ซ",
        "ETDA", "สมอ.", "อย.", "แพลตฟอร์ม", "กขค.", "แข่งขันทางการค้า"
    ],
    "SciTech": [
        "AI", "เอไอ", "ปัญญาประดิษฐ์", "โมเดล", "ออนซ์", "ONNX", "BERT", "Transformer",
        "Google", "Android", "ฟีเจอร์", "ความปลอดภัย", "biometric", "สแกนนิ้ว", "ใบหน้า",
        "Lenovo", "เลอโนโว", "GenAI", "Agentic", "คลาวด์", "ดาต้า", "data-driven",
        "ไฮโดรเจน", "hydrogen", "พลังงานทางเลือก", "คาร์บอน", "CCS", "SMR", "Ammonia", "แอมโมเนีย",
        "เทคโนโลยี", "นวัตกรรม", "โครงสร้างพื้นฐาน", "อินฟรา", "PC", "AI PC"
    ],
    "World": [
        "รัสเซีย", "ยูเครน", "เคียฟ", "ปูติน", "เซเลนสกี", "สงคราม", "หยุดยิง",
        "จีน", "กัมพูชา", "เมียนมา", "อาเซียน", "ต่างประเทศ",
        "ประธานาธิบดี", "ทรัมป์", "รัฐบาล", "ตำรวจ", "ศาล", "จับกุม", "คดี", "ประหารชีวิต",
        "แก๊งคอลเซ็นเตอร์", "ฉ้อโกง", "ข้ามชาติ", "ผู้ต้องสงสัย"
    ]
}

def keyword_score(text: str):
    """
    ให้คะแนนแต่ละหมวดจาก keyword
    - match แบบ substring (ง่าย/เร็ว/พอใช้)
    """
    t = (text or "").lower()
    scores = {k: 0 for k in KEYWORDS.keys()}
    hits = {k: [] for k in KEYWORDS.keys()}

    for label, kws in KEYWORDS.items():
        for kw in kws:
            if kw.lower() in t:
                scores[label] += 1
                hits[label].append(kw)
    return scores, hits


def apply_hybrid_override(text: str, model_pred: str, probs: np.ndarray, id2label: dict):
    """
    ตัดสินใจ override โดยดู keyword score + ความมั่นใจโมเดล
    หลักคิด:
    - ถ้า keyword ของหมวดหนึ่งแรงมาก (>=2) และ "ชนะ" หมวดอื่นชัดเจน → override
    - แต่ถ้าโมเดลมั่นใจจัด (>=0.985) จะยอมโมเดล ยกเว้น keyword ชนะขาด (>=4)
    """
    scores, hits = keyword_score(text)

    # เรียงหมวดตามคะแนน keyword
    best_kw_label = max(scores, key=lambda k: scores[k])
    best_kw_score = scores[best_kw_label]

    # โมเดลทำนายอะไร + มั่นใจแค่ไหน
    model_pred = normalize_label_name(model_pred)
    model_conf = float(max(probs))

    # ถ้า keyword แทบไม่มี อย่าไปยุ่ง
    if best_kw_score < 2:
        return model_pred, {"override": False, "reason": "weak_keywords", "scores": scores, "hits": hits}

    # ถ้าโมเดลมั่นใจมากๆ ให้ยอมโมเดลก่อน (กันการ override มั่ว)
    if model_conf >= 0.985 and best_kw_score < 4:
        return model_pred, {"override": False, "reason": "model_very_confident", "scores": scores, "hits": hits}

    # ถ้า keyword ชนะหมวดอื่นชัด (คะแนนมากกว่าอันดับ 2 อย่างน้อย 1)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_score = sorted_scores[0]
    top2_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

    if top1_score >= 2 and (top1_score - top2_score) >= 1:
        return top1_label, {"override": True, "reason": "keyword_majority", "scores": scores, "hits": hits}

    # ถ้าคะแนนสูสี ไม่ override
    return model_pred, {"override": False, "reason": "keywords_tie", "scores": scores, "hits": hits}


# ============================================================================
# Load Model from Hugging Face Hub (ONNX)
# ============================================================================
def load_models():
    global tokenizer, session, ID2LABEL, LABEL2ID

    try:
        print(f"📥 Downloading model from {REPO_ID}...")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="model",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"   ✅ Downloaded to {LOCAL_DIR}")

        # Load label mapping from config if available
        config_path = os.path.join(LOCAL_DIR, "config.json")
        loaded_map = load_label_mapping_from_config(config_path)
        if loaded_map:
            print(f"🏷️  Loaded label mapping from config.json: {ID2LABEL}")
        else:
            print(f"🏷️  Using fallback label mapping: {ID2LABEL}")

        # Load tokenizer
        tokenizer_path = os.path.join(LOCAL_DIR, "tokenizer")
        print(f"📥 Loading tokenizer from {tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

        # Load ONNX model
        onnx_path = os.path.join(LOCAL_DIR, "model.onnx")
        print(f"📥 Loading ONNX model from {onnx_path}...")
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        # Print model IO names to avoid mismatch issues
        try:
            in_names = [i.name for i in session.get_inputs()]
            out_names = [o.name for o in session.get_outputs()]
            print(f"🔌 ONNX Inputs: {in_names}")
            print(f"🔌 ONNX Outputs: {out_names}")
        except Exception:
            pass

        print("✅ ONNX model loaded successfully!")
        return True

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        tokenizer = None
        session = None
        return False


# Load models at startup
load_models()

# ============================================================================
# API Endpoints
# ============================================================================
@app.route("/")
def home():
    return jsonify({
        "message": "Thai News Topic Classifier API",
        "version": model_info["version"],
        "model": "ONNX Runtime",
        "endpoints": ["GET /health", "GET /model/info", "POST /predict"]
    })


@app.route("/health", methods=["GET"])
def health():
    model_loaded = tokenizer is not None and session is not None
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "model_type": "ONNX Runtime" if model_loaded else None,
        "provider": "CPUExecutionProvider" if model_loaded else None
    }), (200 if model_loaded else 503)


@app.route("/model/info", methods=["GET"])
def get_model_info():
    if tokenizer is None or session is None:
        return jsonify({"error": "Model not loaded"}), 503

    info = model_info.copy()
    info["vocabulary_size"] = len(tokenizer)
    info["model_loaded"] = True
    info["id2label"] = ID2LABEL
    return jsonify(info)


@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    if tokenizer is None or session is None:
        return jsonify({"error": "Model not loaded", "message": "กรุณารอโมเดลโหลดเสร็จ"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No data provided", "message": "กรุณาส่ง JSON body"}), 400

    headline = data.get("headline", "")
    body = data.get("body", "")

    if not headline and not body:
        return jsonify({"error": "Missing required fields", "message": "กรุณากรอก headline หรือ body อย่างน้อย 1 อย่าง"}), 400

    # Combine + preprocess
    text = preprocess_text(f"{headline} {body}".strip())

    # Tokenize: use fixed padding for ONNX stability
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"  # ✅ FIX: stable shape for ONNX
    )

    # Prepare feed dict by matching model input names
    feed = {}
    input_names = [i.name for i in session.get_inputs()]
    if "input_ids" in input_names:
        feed["input_ids"] = inputs["input_ids"]
    if "attention_mask" in input_names:
        feed["attention_mask"] = inputs["attention_mask"]
    # Some exports may include token_type_ids (mostly BERT, not RoBERTa)
    if "token_type_ids" in input_names and "token_type_ids" in inputs:
        feed["token_type_ids"] = inputs["token_type_ids"]

    # Run inference
    outputs = session.run(None, feed)
    raw = outputs[0]  # usually logits
    raw = np.array(raw)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    probs = ensure_probs(raw)[0]
    predicted_id = int(np.argmax(probs))
    predicted_label = ID2LABEL.get(predicted_id, str(predicted_id))
    confidence = float(probs[predicted_id])

    # ✅ Hybrid override
    hybrid_label, hybrid_meta = apply_hybrid_override(
        text=text,
        model_pred=predicted_label,
        probs=probs,
        id2label=ID2LABEL
    )

    final_label = hybrid_label
    final_conf = confidence if final_label == predicted_label else float(probs[LABEL2ID.get(final_label, predicted_id)])

    prob_dict = {ID2LABEL.get(i, str(i)): float(probs[i]) for i in range(len(probs))}

    latency_ms = round((time.time() - start_time) * 1000, 2)

    resp = {
        "label": final_label,
        "confidence": final_conf,
        "probabilities": prob_dict,
        "latency_ms": latency_ms,
        "model_version": model_info["version"],
        "model_type": "ONNX Runtime",
        "hybrid": hybrid_meta,  # ✅ เพิ่ม
        "input": {
            "headline": headline[:100] + "..." if len(headline) > 100 else headline,
            "body": body[:200] + "..." if len(body) > 200 else body
        }
    }

    if DEBUG_PREDICT:
        resp["_debug"] = {
            "id2label": ID2LABEL,
            "predicted_id": predicted_id,
            "raw_output_first_row": [float(x) for x in raw[0].tolist()],
            "text_preview": text[:240]
        }

    return jsonify(resp)


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🇹🇭 Thai News Topic Classifier API")
    print(f"   Model: ONNX Runtime (v{model_info['version']})")
    print("=" * 60)

    if tokenizer is not None and session is not None:
        print(f"   Classes: {list(ID2LABEL.values())}")
        print(f"   Vocabulary size: {len(tokenizer)}")
    else:
        print("   ⚠️ Model not loaded!")

    print("\n📡 Starting server...")
    print("   URL: http://localhost:5001")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5001, debug=True)
