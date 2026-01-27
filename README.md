# 🇹🇭 Thai News Topic Classifier

ระบบจำแนกหมวดหมู่ข่าวภาษาไทย (Thai News Topic Classification)

## 📋 ภาพรวมโปรเจค

| Component | Technology | Description |
|-----------|------------|-------------|
| **Model** | TF-IDF + Logistic Regression | Baseline ML Model |
| **Backend** | Python Flask | REST API |
| **Frontend** | Vite React | Web UI |

## 📊 ผลลัพธ์

- **Accuracy:** 100%
- **Macro-F1:** 1.0
- **Classes:** Business, SciTech, World

## 🚀 Quick Start

### 1. Train Model (Optional)

```bash
cd model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

### 2. Run Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 3. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Open Browser

- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:5000

## 📁 โครงสร้างโปรเจค

```
thai-news-topic-classifier/
├── model/                    # Training scripts
│   ├── train_model.py
│   ├── requirements.txt
│   └── output/               # Trained models
├── backend/                  # Flask API
│   ├── app.py
│   ├── requirements.txt
│   └── models/               # Model files
├── frontend/                 # Vite React
│   ├── src/
│   └── package.json
└── REPORT.md                 # รายงานผลการทดลอง
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Model information |
| POST | `/predict` | Predict topic |

## 📖 รายละเอียดเพิ่มเติม

- [Backend README](./backend/README.md)
- [Frontend README](./frontend/README.md)
- [รายงานผลการทดลอง](./REPORT.md)
