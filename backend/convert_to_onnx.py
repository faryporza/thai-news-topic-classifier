#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert WangchanBERTa Model (Local safetensors) to ONNX
======================================================
โหลดโมเดลจากไฟล์ local /models/model.safetensors
แล้วแปลงเป็น ONNX สำหรับ inference ที่เร็วขึ้น
"""

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================================
# Configuration
# ============================================================================
MODEL_DIR = Path("models")          # โฟลเดอร์ที่มี model.safetensors
OUTPUT_DIR = Path("onnx_model")
ONNX_PATH = OUTPUT_DIR / "model.onnx"
MAX_LENGTH = 256

# ============================================================================
def convert_to_onnx():
    print("=" * 60)
    print("🔄 Converting WangchanBERTa (LOCAL) to ONNX")
    print("=" * 60)

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"❌ Model directory not found: {MODEL_DIR}")

    if not (MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError("❌ model.safetensors not found in /models")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Load tokenizer & model FROM LOCAL PATH
    # ------------------------------------------------------------------------
    print(f"\n📥 Loading tokenizer from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    print(f"📥 Loading model from: {MODEL_DIR}/model.safetensors")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float32,
        local_files_only=True
    )
    model.eval()

    print("   ✅ Model loaded")
    print(f"   - Num labels: {model.config.num_labels}")
    print(f"   - Labels: {model.config.id2label}")

    # ------------------------------------------------------------------------
    # Save tokenizer (for ONNX runtime usage)
    # ------------------------------------------------------------------------
    tokenizer_path = OUTPUT_DIR / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    print(f"\n💾 Tokenizer saved to: {tokenizer_path}")

    # ------------------------------------------------------------------------
    # Dummy input
    # ------------------------------------------------------------------------
    dummy_text = "ตลาดหุ้นไทยปิดบวก"
    dummy = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    # ------------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------------
    print(f"\n🚀 Exporting ONNX → {ONNX_PATH}")

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(ONNX_PATH),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"}
        },
        do_constant_folding=True,
        opset_version=18
    )

    size_mb = ONNX_PATH.stat().st_size / (1024 * 1024)
    print(f"   ✅ ONNX export complete ({size_mb:.2f} MB)")

    # ------------------------------------------------------------------------
    # Verify ONNX
    # ------------------------------------------------------------------------
    print("\n🔍 Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(str(ONNX_PATH))
        onnx.checker.check_model(onnx_model)
        print("   ✅ ONNX model valid")
    except Exception as e:
        print(f"   ❌ ONNX verification failed: {e}")

    # ------------------------------------------------------------------------
    # Test inference
    # ------------------------------------------------------------------------
    print("\n🧪 Testing ONNX inference...")
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(str(ONNX_PATH))

        text = "บริษัทเทคโนโลยีประกาศผลประกอบการ"
        inputs = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )

        logits = session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            }
        )[0]

        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        pred_id = probs.argmax(axis=1)[0]
        label = model.config.id2label[pred_id]

        print(f"   Text: {text}")
        print(f"   Prediction: {label} ({probs[0][pred_id]:.2%})")
        print("   ✅ Inference OK")

    except Exception as e:
        print(f"   ⚠️ ONNX Runtime test skipped: {e}")

    print("\n" + "=" * 60)
    print("🎉 DONE")
    print("=" * 60)


if __name__ == "__main__":
    convert_to_onnx()
