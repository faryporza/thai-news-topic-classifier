# 🇹🇭 Thai News Topic Classifier - Backend API

Flask API สำหรับจำแนกหมวดหมู่ข่าวภาษาไทย

---

## 🆕 Model Update (v2.0)

| Model | Status | Accuracy |
|-------|--------|----------|
| **WangchanBERTa** | ✅ แนะนำ | ~92-97% |
| TF-IDF + Logistic Regression | Legacy | ~85-90% |

### ทำไมถึงเปลี่ยนมาใช้ WangchanBERTa?

1. **Contextual Understanding**: เข้าใจความหมายเชิงบริบท
   - คำว่า "ตลาด" ในข่าวหุ้น vs ข่าวต่างประเทศ → BERT แยกได้
   
2. **Mixed Signal Handling**: จัดการข่าวที่มีหลายสัญญาณได้ดี
   - ข่าว Business ที่พูดถึง AI → TF-IDF สับสน, BERT ทำนายถูก
   
3. **Robust to Typo/Noise**: ทนต่อตัวสะกดผิด
   - TF-IDF ไม่รู้จักคำที่สะกดผิด, BERT เข้าใจได้

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | ตรวจสอบสถานะ API |
| GET | `/model/info` | ข้อมูลโมเดล |
| POST | `/predict` | ทำนายหมวดหมู่ข่าว |

---

## 🚀 วิธีการรัน

### 1. สร้าง Virtual Environment

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 2. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 3. ตรวจสอบ Model Files

**Option A: WangchanBERTa (แนะนำ)**

ต้องมี folder `backend/models/wangchanberta_model/` ที่มีไฟล์:
- `config.json`
- `model.safetensors`
- `tokenizer_config.json`
- `tokenizer.json`
- `special_tokens_map.json`

**Option B: TF-IDF (Legacy)**

ต้องมีไฟล์ใน `backend/models/`:
- `tfidf_vectorizer.joblib`
- `logistic_regression_model.joblib`

### 4. รัน Server

```bash
python app.py
```

Server จะรันที่ http://localhost:5000

---

## 📋 API Usage

### GET /health

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-29T10:00:00",
  "model_loaded": true,
  "model_type": "WangchanBERTa"
}
```

### GET /model/info

```bash
curl http://localhost:5000/model/info
```

Response:
```json
{
  "name": "Thai News Topic Classifier",
  "version": "2.0.0",
  "algorithm": "WangchanBERTa (airesearch/wangchanberta-base-att-spm-uncased)",
  "classes": ["Business", "SciTech", "World"],
  "parameters": "~110 million"
}
```

### POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "บริษัทเทคโนโลยีเปิดตัวผลิตภัณฑ์ใหม่",
    "body": "บริษัทชั้นนำด้าน AI ประกาศเปิดตัวระบบ..."
  }'
```

Response:
```json
{
  "label": "SciTech",
  "confidence": 0.95,
  "probabilities": {
    "Business": 0.03,
    "SciTech": 0.95,
    "World": 0.02
  },
  "latency_ms": 45.2,
  "model_version": "2.0.0",
  "model_type": "WangchanBERTa"
}
```

---

## 🐳 Production Deployment

```bash
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

> ⚠️ **Note**: สำหรับ WangchanBERTa ควรใช้ GPU server เพื่อ inference ที่รวดเร็ว

---

## 📊 Performance Comparison

| Metric | TF-IDF + LR | WangchanBERTa |
|--------|-------------|---------------|
| Accuracy | ~85-90% | ~92-97% |
| Inference Time | ~5ms | ~30-50ms (CPU) / ~10ms (GPU) |
| Model Size | ~10 MB | ~400 MB |

---

## 📝 Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-29 | เพิ่ม WangchanBERTa model support |
| 1.0.0 | 2026-01-27 | Initial release with TF-IDF + LR |
