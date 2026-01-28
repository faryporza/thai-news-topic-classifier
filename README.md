# 🇹🇭 Thai News Topic Classifier

ระบบจำแนกหมวดหมู่ข่าวภาษาไทย (Thai News Topic Classification) พัฒนาด้วย Machine Learning และ Deploy เป็น Web Application

---

## 📋 ภาพรวมโปรเจค

| Component | Technology | Description |
|-----------|------------|-------------|
| **Model** | TF-IDF + Logistic Regression | Baseline ML Model |
| **Backend** | Python Flask + Gunicorn | REST API |
| **Frontend** | Vite + React + Tailwind CSS | Web UI |

## 📊 ผลลัพธ์การ Train

| Metric | Score |
|--------|-------|
| **Accuracy** | 100% |
| **Macro-F1** | 1.0 |
| **Classes** | Business, SciTech, World |

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd thai-news-topic-classifier
```

### 2. Train Model (Optional - models พร้อมใช้แล้ว)

```bash
cd model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

### 3. Run Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

### 4. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

### 5. Open Browser

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:5001

---

## 📁 โครงสร้างโปรเจค

```
thai-news-topic-classifier/
├── model/                          # Training scripts
│   ├── train_model.py              # Script train โมเดล 5 ขั้นตอน
│   ├── requirements.txt            # Python dependencies
│   ├── 12.agnews_thai_train_easy.csv  # Dataset
│   └── output/                     # Trained models
│       ├── tfidf_vectorizer.joblib
│       ├── logistic_regression_model.joblib
│       └── confusion_matrix.png
│
├── backend/                        # Flask API
│   ├── app.py                      # API endpoints
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # API documentation
│   └── models/                     # Model files (copy from model/output)
│       ├── tfidf_vectorizer.joblib
│       └── logistic_regression_model.joblib
│
├── frontend/                       # Vite React
│   ├── src/
│   │   ├── App.jsx                 # Main component
│   │   ├── index.css               # Tailwind CSS
│   │   └── data/                   # JSON data files
│   │       ├── sampleNews.json     # ตัวอย่างข่าว (17 ข่าว)
│   │       └── misclassifiedExamples.json  # Error examples (12 ตัวอย่าง)
│   ├── .env                        # Environment variables
│   ├── package.json
│   └── README.md
│
├── README.md                       # This file
└── REPORT.md                       # รายงานผลการทดลอง
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | ตรวจสอบสถานะ API |
| GET | `/model/info` | ข้อมูลโมเดล (version, classes, vocabulary size) |
| POST | `/predict` | ทำนายหมวดหมู่ข่าว |

### Example: POST /predict

**Request:**
```json
{
  "headline": "ตลาดหุ้นไทยปิดบวก 15 จุด",
  "body": "ดัชนีตลาดหลักทรัพย์แห่งประเทศไทยปิดบวก..."
}
```

**Response:**
```json
{
  "label": "Business",
  "confidence": 0.95,
  "probabilities": {
    "Business": 0.95,
    "SciTech": 0.03,
    "World": 0.02
  },
  "latency_ms": 12.5,
  "model_version": "1.0.0"
}
```

---

## 🎨 Web Application Features

### หน้าหลัก (Prediction Page)
- ✅ ช่องกรอก Headline และ Body
- ✅ ปุ่ม **Try Example** - สุ่มข่าวตัวอย่าง
- ✅ ปุ่ม **Predict** - ทำนายหมวดหมู่
- ✅ แสดง **Predicted Label** และ **Confidence Score**
- ✅ **Probability Bar Chart** - กราฟแท่งความน่าจะเป็น
- ✅ **Latency** และ **Model Version**

### หน้า Error Analysis
- ✅ แสดงตัวอย่างที่โมเดลทำนายผิด (12 ตัวอย่าง)
- ✅ แสดง Actual vs Predicted Label
- ✅ การวิเคราะห์สาเหตุ (Mixed Signal, Domain Shift)
- ✅ ข้อเสนอแนะในการปรับปรุง

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **ML Framework** | scikit-learn |
| **Backend** | Flask, Gunicorn |
| **Frontend** | Vite, React 18, Tailwind CSS |
| **Icons** | Lucide React |
| **Model Serialization** | joblib |

---

## 📖 Documentation

- [Backend README](./backend/README.md) - API documentation
- [Frontend README](./frontend/README.md) - Frontend features & setup
- [REPORT.md](./REPORT.md) - รายงานผลการทดลอง (2-4 หน้า)

---

## 👥 Contributors

| ชื่อ-สกุล | รหัสนักศึกษา |
|----------|-------------|
| นาย อภิรักษ์ เขื่อนคำ | 66021140 |
| นาย สิทธิพล สุขอินทร์ | 66024941 |
| นาย ประขรรค์ จันสุกปุก | 66020879 |
| นาย ธนกฤต ชูเชิด | 66025694 |
| นาย พายุ พันธ์วงศ์ | 66020925 |

---

## 📝 License

Thai News Topic Classifier © 2026 | Machine Learning & Data Science Course
