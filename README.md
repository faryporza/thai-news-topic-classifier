# 🇹🇭 Thai News Topic Classifier

ระบบจำแนกหมวดหมู่ข่าวภาษาไทย (Thai News Topic Classification) พัฒนาด้วย Machine Learning และ Deploy เป็น Web Application

---

## 🆕 Model Update v2.0 - WangchanBERTa

> **เปลี่ยนจาก TF-IDF + Logistic Regression มาใช้ WangchanBERTa เพื่อความแม่นยำที่สูงขึ้น**

### ทำไมถึงเปลี่ยน?

| ปัญหาของ TF-IDF | WangchanBERTa แก้ได้อย่างไร |
|-----------------|---------------------------|
| ❌ ไม่เข้าใจบริบท (คำว่า "ตลาด" ในข่าวหุ้น vs ข่าวต่างประเทศ ถูกมองเหมือนกัน) | ✅ **Contextual Understanding** - เข้าใจความหมายตามบริบท |
| ❌ สับสนกับ Mixed Signal (ข่าว Business ที่พูดถึง AI) | ✅ **Mixed Signal Handling** - แยกแยะประเด็นหลักได้ |
| ❌ ไม่ทน Typo (คำสะกดผิดไม่รู้จัก) | ✅ **Robust to Noise** - เข้าใจแม้สะกดผิด |
| ❌ OOV Problem (คำใหม่ถูกละเลย) | ✅ **Subword Tokenization** - รู้จักคำใหม่ได้ |

### Performance Comparison

| Model | Accuracy | Macro-F1 | Status |
|-------|----------|----------|--------|
| TF-IDF + Logistic Regression | ~85-90% | ~0.85-0.90 | Baseline |
| **WangchanBERTa** (Production) | **100%** ✅ | **1.0000** | 🟢 Active |

### 🧠 ทำไม WangchanBERTa ดีกว่าโมเดลทั่วไป?

| จุดเด่น | คำอธิบาย |
|--------|---------|
| **SentencePiece Tokenizer** | ตัดคำเป็น subword ไม่ต้องพึ่ง dictionary → รับมือกับคำใหม่, คำทับศัพท์ |
| **ฝึกจากข้อมูลไทยจริง** | ข่าว, บทความ, เอกสารภาษาไทย → เข้าใจบริบทข่าวไทย |
| **โครงสร้าง RoBERTa** | Dynamic Masking, ฝึกนานกว่า → เข้าใจประโยคยาวได้ดี |

> 💡 **สรุป:** "WangchanBERTa ถูกออกแบบมาเพื่อเข้าใจภาษาไทยตั้งแต่ระดับการตัดคำ จนถึงบริบทของข่าวจริง ๆ"

### ⚡ ONNX Runtime

โมเดลถูกแปลงเป็น ONNX เพื่อ:
- Inference เร็วขึ้น **2-3 เท่า**
- ไม่ต้องใช้ GPU
- Deploy ง่ายบน Cloud
- **Hybrid Post-processing**: ผสาน Rule-based Logic เข้ากับ AI เพื่อแก้ไขกรณี Mixed Signal

### 🧠 Hybrid Post-processing (v2.1)

ระบบใช้การทำงานร่วมกันระหว่าง **WangchanBERTa** และ **Keyword Scoring**:

1. **Model Prediction**: ให้ AI ทำนายผลเบื้องต้น
2. **Keyword Match**: ตรวจจับคำสำคัญในข่าว (เช่น "หุ้น", "AI", "สงคราม")
3. **Hybrid Logic**:
   - ถ้า AI มั่นใจสูง (> 98.5%) → เชื่อ AI
   - ถ้า AI ลังเล และเจอ Keyword ชัดเจน → เชื่อ Keyword (Rule Override)
   - ช่วยแก้ปัญหาข่าวที่มีความกำกวม (เช่น ข่าว Tech ที่มีคำว่า "หุ้น") ได้อย่างแม่นยำ

## 📋 ภาพรวมโปรเจค

| Component | Technology | Description |
|-----------|------------|-------------|
| **Model** | WangchanBERTa / TF-IDF + LR | Thai Text Classification |
| **Backend** | Python Flask + Gunicorn | REST API |
| **Frontend** | Vite + React + Tailwind CSS | Web UI |

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd thai-news-topic-classifier
```

### 2. Train Model

**Option A: WangchanBERTa (แนะนำ - ความแม่นยำสูง)**

```bash
cd model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements_bert.txt
python train_wangchanberta.py
```

> ⚠️ ต้องใช้ GPU สำหรับ training ที่รวดเร็ว

**Option B: TF-IDF + Logistic Regression (Baseline - เร็ว)**

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
├── model/                              # Training scripts
│   ├── train_wangchanberta.py          # 🆕 WangchanBERTa training
│   ├── train_model.py                  # TF-IDF + LR training
│   ├── requirements_bert.txt           # BERT dependencies
│   ├── requirements.txt                # TF-IDF dependencies
│   ├── 12.agnews_thai_train_easy.csv   # Dataset
│   ├── output/                         # TF-IDF models
│   │   ├── tfidf_vectorizer.joblib
│   │   └── logistic_regression_model.joblib
│   └── output_bert/                    # 🆕 BERT models
│       └── wangchanberta_model/
│
├── backend/                            # Flask API
│   ├── app.py                          # API endpoints
│   ├── requirements.txt                # Python dependencies
│   ├── README.md                       # API documentation
│   └── models/                         # Model files
│
├── frontend/                           # Vite React
│   ├── src/
│   │   ├── App.jsx                     # Main component
│   │   ├── index.css                   # Tailwind CSS
│   │   └── data/                       # JSON data files
│   ├── .env
│   ├── package.json
│   └── README.md
│
├── README.md                           # This file
└── REPORT.md                           # รายงานผลการทดลอง
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | ตรวจสอบสถานะ API |
| GET | `/model/info` | ข้อมูลโมเดล (version, type, classes) |
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
  "model_version": "2.0.0",
  "model_type": "WangchanBERTa"
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
- ✅ แสดงตัวอย่างที่โมเดลทำนายผิด
- ✅ แสดง Actual vs Predicted Label
- ✅ การวิเคราะห์สาเหตุ (Mixed Signal, Domain Shift)
- ✅ ข้อเสนอแนะในการปรับปรุง

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **ML Framework** | PyTorch, Transformers, scikit-learn |
| **Model** | WangchanBERTa (airesearch) |
| **Backend** | Flask, Gunicorn |
| **Frontend** | Vite, React 18, Tailwind CSS |
| **Icons** | Lucide React |

---

## 📖 Documentation

- [Model README](./model/README.md) - Training documentation
- [Backend README](./backend/README.md) - API documentation
- [Frontend README](./frontend/README.md) - Frontend features & setup
- [REPORT.md](./REPORT.md) - รายงานผลการทดลอง

---

## 📊 Model Comparison

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Aspect              │ TF-IDF + LR          │ WangchanBERTa        │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Accuracy (expected) │ ~85-90%              │ ~92-97%              │
│ Context Understanding│ ❌ No               │ ✅ Yes               │
│ Mixed Signal        │ ❌ Struggles         │ ✅ Handles well      │
│ Typo Tolerance      │ ❌ Low               │ ✅ High              │
│ Training Speed      │ ✅ Fast (seconds)    │ ❌ Slow (minutes)    │
│ Inference Speed     │ ✅ Very Fast (~5ms)  │ ⚠️ Moderate (~30ms)  │
│ Model Size          │ ✅ Small (~10 MB)    │ ❌ Large (~400 MB)   │
│ GPU Required        │ ❌ No                │ ⚠️ Recommended       │
└─────────────────────┴──────────────────────┴──────────────────────┘

🎯 Recommendation:
- Production with high accuracy → WangchanBERTa
- Quick prototyping / low resource → TF-IDF + Logistic Regression
```

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

## 📝 Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.1.0 | 2026-01-31 | 🧠 เพิ่ม Hybrid Post-processing (Rule-based Override) แก้ไข Mixed Signal |
| 2.0.0 | 2026-01-30 | 🎉 WangchanBERTa ได้ **100% Accuracy** - Deploy เป็น Production |
| 1.5.0 | 2026-01-29 | เพิ่ม WangchanBERTa model |
| 1.0.0 | 2026-01-27 | Initial release with TF-IDF + Logistic Regression |

---

## 📝 License

Thai News Topic Classifier © 2026 | Machine Learning & Data Science Course
