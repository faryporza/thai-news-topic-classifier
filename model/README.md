# 🇹🇭 Thai News Topic Classifier

โปรแกรมสำหรับจำแนกหมวดหมู่ข่าวภาษาไทย (Thai News Topic Classification) โดยใช้ TF-IDF และ Logistic Regression

## 📋 รายละเอียด Dataset

- **ประเภทข้อมูล**: ข่าวภาษาไทย (Thai News Topic Dataset)
- **Input Features**: `headline` (พาดหัว) + `body` (เนื้อหาข่าว)
- **Target Label**: `topic` (SciTech, World, Business)
- **ลักษณะข้อมูล**: train_easy, version clean

## 🚀 วิธีการรัน

### 1. สร้าง Virtual Environment

```bash
# สร้าง venv
python3 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate
```

### 2. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 3. รัน Training

```bash
python train_model.py
```

### 4. Output Files

หลังจากรันเสร็จจะได้ไฟล์ใน folder `output/`:
- `tfidf_vectorizer.joblib` - TF-IDF Vectorizer
- `logistic_regression_model.joblib` - Trained Model
- `confusion_matrix.png` - Confusion Matrix Plot
- `misclassified_samples.csv` - ตัวอย่างที่ทำนายผิด

## 📊 การ Deploy / ใช้งานโมเดล

```python
import joblib

# โหลดโมเดล
vectorizer = joblib.load('output/tfidf_vectorizer.joblib')
model = joblib.load('output/logistic_regression_model.joblib')

# ทำนาย
def predict_topic(headline: str, body: str) -> str:
    text = headline + ' ' + body
    X = vectorizer.transform([text])
    prediction = model.predict(X)
    return prediction[0]

# ตัวอย่างการใช้งาน
topic = predict_topic(
    headline="บริษัทเทคโนโลยีเปิดตัวผลิตภัณฑ์ใหม่",
    body="บริษัทชั้นนำด้าน AI ประกาศเปิดตัวระบบ..."
)
print(f"Predicted topic: {topic}")
```

## 📁 โครงสร้างโปรเจค

```
model/
├── 12.agnews_thai_train_easy.csv  # Dataset
├── train_model.py                  # Training script
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── output/                         # Output folder
    ├── tfidf_vectorizer.joblib
    ├── logistic_regression_model.joblib
    ├── confusion_matrix.png
    └── misclassified_samples.csv
```

## ⚙️ Model Configuration

| Component | Configuration |
|-----------|---------------|
| Feature Extraction | TF-IDF (word-level, unigram + bigram) |
| Model | Logistic Regression |
| class_weight | `balanced` |
| max_features | 10,000 |
| Test Size | 20% |

## 📈 Expected Performance

- **Accuracy**: ~85-90%
- **Macro-F1**: ~0.85-0.90

## 🔧 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib
- seaborn
