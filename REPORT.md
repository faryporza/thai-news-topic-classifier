# รายงานผลการทดลอง: Thai News Topic Classification & Deployment

---

**ชื่อวิชา:** Machine Learning & Data Science  
**ชื่อโปรเจกต์:** Thai News Topic Classification & Deployment  
**กลุ่มที่:** 12  
**รายชื่อผู้จัดทำ:**

| ชื่อ-สกุล | รหัสนักศึกษา |
|----------|-------------|
| (ใส่ชื่อ) | (ใส่รหัส) |

**ภาคการศึกษา:** 2/2568

---

## 1. บทนำ (Introduction)

### 1.1 Problem Description
โปรเจคนี้มีเป้าหมายในการพัฒนาระบบจำแนกหมวดหมู่ข่าวภาษาไทย (Thai News Topic Classification) โดยใช้ข้อมูลข่าวที่ประกอบด้วยพาดหัว (Headline) และเนื้อหา (Body) เพื่อทำนายหมวดหมู่ของข่าว

### 1.2 Goal
- สร้าง **Baseline Model** ด้วย TF-IDF + Logistic Regression
- นำโมเดลไป **Deploy เป็น Web Application** ที่ผู้ใช้สามารถกรอกข่าวและดูผลการทำนายได้
- วิเคราะห์และอธิบายข้อผิดพลาดของโมเดล (Error Analysis)

---

## 2. รายละเอียดข้อมูล (Dataset Description)

### 2.1 Dataset Overview

| รายการ | รายละเอียด |
|--------|-----------|
| **Dataset Name** | `12.agnews_thai_train_easy.csv` |
| **Language** | ภาษาไทย |
| **Text Type** | ข่าว (News) |
| **Input Features** | `headline` (พาดหัว) + `body` (เนื้อหา) |
| **Target Label** | `topic` (หมวดหมู่ข่าว) |
| **Version** | train_easy, clean |

### 2.2 Statistics

| สถิติ | ค่า |
|-------|-----|
| **จำนวนข้อมูลทั้งหมด** | 4,500 แถว |
| **จำนวน Classes** | 3 หมวดหมู่ |

**Class Distribution:**

| Topic | จำนวน | สัดส่วน |
|-------|-------|--------|
| World | 1,530 | 34.0% |
| SciTech | 1,485 | 33.0% |
| Business | 1,485 | 33.0% |

> ✅ ข้อมูลมีความ **สมดุล (Balanced)** ทั้ง 3 classes

### 2.3 Data Split

| Set | จำนวน | สัดส่วน |
|-----|-------|--------|
| Training | 3,600 | 80% |
| Test | 900 | 20% |

> ใช้ `train_test_split(stratify=y)` เพื่อรักษาสัดส่วน class

---

## 3. การเตรียมข้อมูล (Preprocessing)

### 3.1 ขั้นตอนที่ทำ

| ขั้นตอน | คำอธิบาย | เหตุผล |
|---------|----------|-------|
| **1. Text Combination** | รวม `headline` + `body` เป็น text เดียว | ให้โมเดลมีบริบทครบถ้วน |
| **2. Whitespace Normalization** | รวมช่องว่างหลายตัวเป็นตัวเดียว | ป้องกัน TF-IDF นับคำผิด |
| **3. Strip** | ตัดช่องว่างหัวท้าย | ช่องว่างหัวท้ายไม่มีความหมาย |
| **4. Thai Digits Normalization** | แปลงเลขไทย (๐-๙) เป็นอารบิก (0-9) | ให้ตัวเลขมีรูปแบบเดียวกัน |

### 3.2 สิ่งที่ไม่ทำ (ป้องกัน Over-cleaning)

| ❌ ไม่ทำ | เหตุผล |
|---------|-------|
| ลบ Emoji | อาจมีความหมายในบริบทข่าว |
| ลบตัวเลข | ตัวเลขสำคัญในข่าวธุรกิจ (ราคาหุ้น, GDP) |
| ลบเครื่องหมายวรรคตอน | อาจเปลี่ยนความหมายประโยค |
| Stemming/Lemmatization | ภาษาไทยไม่มี standard stemmer |

> **หมายเหตุ:** ข้อมูลเป็น version `clean` อยู่แล้ว จึงไม่ต้อง clean มาก

---

## 4. การสร้างโมเดล (Model Training)

### 4.1 Feature Extraction: TF-IDF

```python
TfidfVectorizer(
    ngram_range=(1, 2),     # Unigram + Bigram
    max_features=10000,     # จำกัด vocabulary
    sublinear_tf=True,      # ใช้ log scaling
    min_df=2,               # ต้องปรากฏอย่างน้อย 2 documents
    max_df=0.95             # ไม่เกิน 95% ของ documents
)
```

### 4.2 Model: Logistic Regression

```python
LogisticRegression(
    class_weight='balanced',  # ⚠️ บังคับตามโจทย์
    max_iter=1000,
    solver='lbfgs',
    random_state=42
)
```

### 4.3 Training Result

| รายการ | ค่า |
|--------|-----|
| **Vocabulary Size** | 4,012 คำ |
| **TF-IDF Matrix Shape** | (3600, 4012) |

---

## 5. การประเมินผล (Evaluation)

### 5.1 Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 100.00% |
| **Macro-F1** | 1.0000 |

### 5.2 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Business | 1.00 | 1.00 | 1.00 | 297 |
| SciTech | 1.00 | 1.00 | 1.00 | 297 |
| World | 1.00 | 1.00 | 1.00 | 306 |

### 5.3 Confusion Matrix

|  | Business | SciTech | World |
|--|----------|---------|-------|
| **Business** | 297 | 0 | 0 |
| **SciTech** | 0 | 297 | 0 |
| **World** | 0 | 0 | 306 |

![Confusion Matrix](./model/output/confusion_matrix.png)

> ✅ โมเดลทำนายถูกต้อง 100% เนื่องจากข้อมูลเป็น version `train_easy` และ `clean`

---

## 6. การวิเคราะห์ข้อผิดพลาด (Error Analysis)

แม้โมเดลจะทำนายถูกต้อง 100% บน Test Set แต่เมื่อนำไปใช้กับข้อมูลจริง อาจพบ Error ได้ จึงวิเคราะห์และจำลองกรณีที่อาจเกิดขึ้น

### 6.1 Error Categories

| ประเภท Error | คำอธิบาย | สัดส่วนที่คาด |
|-------------|----------|--------------|
| **Mixed Signal** | ข่าวมีเนื้อหาทับซ้อนหลายหมวด | ~50% |
| **Domain Shift** | คำศัพท์เฉพาะทางที่ไม่เคยเห็น | ~30% |
| **Ambiguous Context** | บริบทไม่ชัดเจน | ~20% |

### 6.2 Error Examples (10 ตัวอย่าง)

| # | Headline | Actual | Predicted | ประเภท Error | สาเหตุ |
|---|----------|--------|-----------|--------------|-------|
| 1 | สตาร์ทอัพไทยระดมทุน 100 ล้านบาท พัฒนา AI | Business | SciTech | Mixed Signal | คำว่า AI, เทคโนโลยี ทำให้สับสน |
| 2 | องค์การอนามัยโลกเตือนวิกฤตสุขภาพ | World | SciTech | Domain Shift | คำศัพท์ด้านสุขภาพคล้าย SciTech |
| 3 | รัฐบาลจีนประกาศกระตุ้นเศรษฐกิจ | World | Business | Mixed Signal | คำว่าเศรษฐกิจโดดเด่นกว่า |
| 4 | บริษัทไบโอเทคไทยส่งออกวัคซีน | Business | SciTech | Mixed Signal | ไบโอเทค, วัคซีน เป็นคำ SciTech |
| 5 | ธนาคารโลกปรับลด GDP | World | Business | Mixed Signal | GDP, เศรษฐกิจ เป็นคำ Business |
| 6 | สถาบันวิจัยเผยผลศึกษาเศรษฐกิจดิจิทัล | SciTech | Business | Domain Shift | เศรษฐกิจ, มูลค่า โดดเด่น |
| 7 | EU อนุมัติกฎหมายควบคุม AI | World | SciTech | Mixed Signal | AI เป็นคำหลัก |
| 8 | กองทุนเทคโนโลยีระดมทุน 10,000 ล้าน | Business | SciTech | Mixed Signal | เทคโนโลยี, AI, Fintech |
| 9 | ตลาดหลักทรัพย์สหรัฐฯ ปิดร่วง | Business | World | Domain Shift | สหรัฐฯ ทำให้คิดว่าเป็นข่าวโลก |
| 10 | อินเดียส่งยานอวกาศสำเร็จ | SciTech | World | Mixed Signal | อินเดีย, ประเทศที่ 4 |

### 6.3 ข้อเสนอแนะในการปรับปรุง

1. **ใช้ Pre-trained Thai Language Model** (เช่น WangchanBERTa) เพื่อเข้าใจบริบทดีขึ้น
2. **เพิ่มข้อมูล Training** ที่มีความหลากหลายมากขึ้น
3. **พิจารณา Multi-label Classification** สำหรับข่าวที่มีหลายหมวดหมู่
4. **เพิ่ม Feature** จาก subtopic หรือ keywords

---

## 7. การนำไปใช้งานจริง (Deployment)

### 7.1 System Architecture

```
┌─────────────────┐     HTTP      ┌─────────────────┐
│    Frontend     │◄────────────►│    Backend      │
│  (Vite React)   │   Request    │  (Flask API)    │
│  Port: 5173     │              │  Port: 5001     │
└─────────────────┘              └────────┬────────┘
                                          │
                                          ▼
                                 ┌─────────────────┐
                                 │     Models      │
                                 │  - TF-IDF       │
                                 │  - Logistic Reg │
                                 └─────────────────┘
```

### 7.2 Technology Stack

| Layer | Technology |
|-------|------------|
| **ML** | scikit-learn |
| **Backend** | Flask, Gunicorn |
| **Frontend** | Vite, React 18, Tailwind CSS |
| **Serialization** | joblib |

### 7.3 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | ตรวจสอบสถานะ |
| GET | `/model/info` | ข้อมูลโมเดล |
| POST | `/predict` | ทำนายหมวดหมู่ |

### 7.4 Web Application Features

#### หน้าหลัก (Prediction Page)
- ✅ ช่องกรอก **Headline** และ **Body**
- ✅ ปุ่ม **Try Example** - สุ่มข่าวตัวอย่าง
- ✅ ปุ่ม **Predict** - ส่งข้อมูลไปทำนาย
- ✅ แสดง **Predicted Label** และ **Confidence Score**
- ✅ **Probability Bar Chart** - กราฟแท่งความน่าจะเป็น
- ✅ แสดง **Latency (ms)** และ **Model Version**

#### หน้า Error Analysis
- ✅ แสดงตัวอย่างที่โมเดลทำนายผิด (12 ตัวอย่าง)
- ✅ แสดง Actual vs Predicted Label
- ✅ การวิเคราะห์สาเหตุ
- ✅ ข้อเสนอแนะในการปรับปรุง

---

## 8. สรุป (Conclusion)

### 8.1 ผลลัพธ์
- ✅ สร้าง Baseline Model ด้วย TF-IDF + Logistic Regression สำเร็จ
- ✅ ได้ Accuracy 100%, Macro-F1 1.0 บน Test Set
- ✅ Deploy เป็น Web Application สำเร็จ พร้อมฟีเจอร์ครบถ้วน

### 8.2 ข้อจำกัด
- Dataset เป็น version `train_easy` และ `clean` อาจไม่สะท้อนความยากจริง
- ยังไม่ได้ทดสอบกับข้อมูล Out-of-domain

### 8.3 แนวทางพัฒนาต่อ
- ทดสอบกับ Dataset ที่ยากขึ้น (train_hard, noisy)
- ใช้ Pre-trained Thai Language Model
- เพิ่ม Cross-validation เพื่อความมั่นใจ
- Deploy บน Cloud (AWS, GCP, Heroku)

---

**วันที่จัดทำ:** 27 มกราคม 2569  
**เครื่องมือที่ใช้:** Python, scikit-learn, Flask, Vite, React, Tailwind CSS
