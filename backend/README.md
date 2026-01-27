# 🇹🇭 Thai News Topic Classifier - Backend API

Flask API สำหรับจำแนกหมวดหมู่ข่าวภาษาไทย

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | ตรวจสอบสถานะ API |
| GET | `/model/info` | ข้อมูลโมเดล |
| POST | `/predict` | ทำนายหมวดหมู่ข่าว |

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

ต้องมีไฟล์ใน `backend/models/`:
- `tfidf_vectorizer.joblib`
- `logistic_regression_model.joblib`

### 4. รัน Server

```bash
python app.py
```

Server จะรันที่ http://localhost:5000

## 📋 API Usage

### GET /health

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-27T22:00:00",
  "model_loaded": true
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
  "version": "1.0.0",
  "algorithm": "TF-IDF + Logistic Regression",
  "classes": ["Business", "SciTech", "World"],
  "vocabulary_size": 4012
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
  }
}
```

## 🐳 Production Deployment

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
