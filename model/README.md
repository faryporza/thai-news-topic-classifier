# 🇹🇭 Thai News Topic Classifier - Model Training

โปรแกรมสำหรับจำแนกหมวดหมู่ข่าวภาษาไทย (Thai News Topic Classification)

---

## 🚀 Model Options

| Model | Script | Accuracy | Speed | Status |
|-------|--------|----------|-------|--------|
| **WangchanBERTa** (ใช้งานจริง) | `train_wangchanberta.py` | **100%** ✅ | ⚠️ ช้า (ต้องใช้ GPU) | 🟢 Active |
| TF-IDF + Logistic Regression | `train_model.py` | ~85-90% | ✅ เร็ว | 🔴 Deprecated |

> **✅ โมเดล WangchanBERTa ได้ความแม่นยำ 100% บน Test Set (900 samples)**

---

## 🏆 ทำไมถึงเปลี่ยนมาใช้ WangchanBERTa?

### ❌ ข้อจำกัดของ TF-IDF + Logistic Regression

| ปัญหา | คำอธิบาย | ตัวอย่าง |
|-------|----------|----------|
| **ไม่เข้าใจบริบท** | มองทุกคำเป็นอิสระ ไม่รู้ว่าคำอยู่ในบริบทไหน | คำว่า "ตลาด" ในข่าวหุ้น vs ข่าวต่างประเทศ ถูกมองเหมือนกัน |
| **Mixed Signal** | สับสนเมื่อข่าวมีหลายหมวดหมู่ทับซ้อน | ข่าว Business ที่พูดถึง AI อาจถูกทำนายเป็น SciTech |
| **ไม่ทน Typo** | คำที่สะกดผิดจะไม่รู้จัก | "เทตโนโลยี" → ไม่รู้จัก |
| **OOV Problem** | คำใหม่ที่ไม่เคยเห็นตอน training จะถูกละเลย | - |

### ✅ ข้อดีของ WangchanBERTa

| คุณสมบัติ | คำอธิบาย |
|----------|----------|
| **Contextual Understanding** | เข้าใจความหมายเชิงบริบท - คำเดียวกันต่างบริบทให้ความหมายต่างกัน |
| **Pre-trained Knowledge** | เรียนรู้จาก corpus ภาษาไทยขนาดใหญ่ มีความรู้พื้นฐานอยู่แล้ว |
| **Subword Tokenization** | แยกคำเป็น subword ทำให้รู้จักคำที่ไม่เคยเห็นได้ |
| **Robust to Noise** | ทนต่อ typo และ noise ได้ดีกว่า |

### 📊 ตัวอย่างเปรียบเทียบ

```
ข่าว: "ตลาดหุ้นปิดบวก 15 จุด ท่ามกลางข่าวเทคโนโลยี AI"

TF-IDF:
- พบคำ "ตลาด", "หุ้น" → น่าจะ Business
- พบคำ "เทคโนโลยี", "AI" → น่าจะ SciTech
- ❌ สับสน! ไม่รู้ว่าประเด็นหลักคืออะไร

WangchanBERTa:
- เข้าใจว่า "ตลาดหุ้น" เป็นหัวข้อหลัก
- "เทคโนโลยี AI" เป็นบริบทประกอบ
- ✅ ทำนายถูกว่าเป็น Business
```

---

## 📋 รายละเอียด Dataset

- **ประเภทข้อมูล**: ข่าวภาษาไทย (Thai News Topic Dataset)
- **Input Features**: `headline` (พาดหัว) + `body` (เนื้อหาข่าว)
- **Target Label**: `topic` (SciTech, World, Business)
- **ลักษณะข้อมูล**: train_easy, version clean
- **จำนวน samples**: ~4,500

---

## 🚀 วิธีการรัน

### Option 1: WangchanBERTa (แนะนำ - ความแม่นยำสูง)

#### 1. สร้าง Virtual Environment

```bash
cd model
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

#### 2. ติดตั้ง Dependencies

```bash
pip install -r requirements_bert.txt
```

#### 3. รัน Training

```bash
python train_wangchanberta.py
```

> ⚠️ **Note**: ต้องใช้ GPU สำหรับ training ที่รวดเร็ว (รองรับ CUDA และ Apple Silicon MPS)

---

### Option 2: TF-IDF + Logistic Regression (Baseline - เร็ว)

```bash
cd model
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

---

## 📁 Output Files

### WangchanBERTa (`output_bert/`)
```
output_bert/
├── wangchanberta_model/       # Trained model directory
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   └── ...
├── confusion_matrix_bert.png  # Confusion matrix
└── logs/                      # Training logs
```

### TF-IDF (`output/`)
```
output/
├── tfidf_vectorizer.joblib
├── logistic_regression_model.joblib
├── confusion_matrix.png
└── misclassified_samples.csv
```

---

## 📊 การ Deploy / ใช้งานโมเดล

### WangchanBERTa

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# โหลดโมเดล
model_path = "output_bert/wangchanberta_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_topic(headline: str, body: str) -> dict:
    text = headline + ' ' + body
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        predicted_id = torch.argmax(probs).item()
    
    return {
        "label": model.config.id2label[predicted_id],
        "confidence": probs[predicted_id].item(),
        "probabilities": {
            model.config.id2label[i]: probs[i].item() 
            for i in range(len(probs))
        }
    }

# ตัวอย่างการใช้งาน
result = predict_topic(
    headline="บริษัทเทคโนโลยีเปิดตัวผลิตภัณฑ์ใหม่",
    body="บริษัทชั้นนำด้าน AI ประกาศเปิดตัวระบบ..."
)
print(f"Predicted: {result['label']} (confidence: {result['confidence']:.2%})")
```

### TF-IDF + Logistic Regression

```python
import joblib

vectorizer = joblib.load('output/tfidf_vectorizer.joblib')
model = joblib.load('output/logistic_regression_model.joblib')

def predict_topic(headline: str, body: str) -> str:
    text = headline + ' ' + body
    X = vectorizer.transform([text])
    return model.predict(X)[0]
```

---

## ⚙️ Model Configuration Comparison

| Configuration | TF-IDF + LR | WangchanBERTa |
|--------------|-------------|---------------|
| **Feature Extraction** | TF-IDF (unigram + bigram) | BERT Tokenizer (Subword) |
| **Model** | Logistic Regression | Transformer (BERT) |
| **Max Features/Length** | 10,000 features | 256 tokens |
| **Parameters** | ~40K | ~110M |
| **Training Time** | ~10 seconds | ~10-30 minutes |
| **GPU Required** | ❌ No | ✅ Recommended |

---

## 📈 Actual Performance (Test Set: 900 samples)

| Model | Accuracy | Macro-F1 | Status |
|-------|----------|----------|--------|
| TF-IDF + Logistic Regression | ~85-90% | ~0.85-0.90 | Baseline |
| **WangchanBERTa** | **100%** ✅ | **1.0000** | **Production** |

> 🎯 **Perfect Score!** WangchanBERTa ทายถูก 900/900 samples (Business: 297, SciTech: 297, World: 306)

---

## 🔧 Requirements

### For WangchanBERTa (`requirements_bert.txt`)
```
torch>=2.0.0
transformers>=4.30.0
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### For TF-IDF (`requirements.txt`)
```
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
```

---

## 📝 Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-29 | 2.0.0 | เพิ่ม WangchanBERTa model (แนะนำ) |
| 2026-01-27 | 1.0.0 | Initial release with TF-IDF + Logistic Regression |
