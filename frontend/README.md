# 🇹🇭 Thai News Topic Classifier - Frontend

Vite React Frontend สำหรับระบบจำแนกหมวดหมู่ข่าวภาษาไทย

## � ภาพรวม

Frontend นี้พัฒนาด้วย **Vite + React + Tailwind CSS** เพื่อให้ผู้ใช้สามารถกรอกข่าวภาษาไทยและรับผลการทำนายหมวดหมู่จาก Backend API

---

## 🎯 ความสามารถของระบบ (Features)

### 1. หน้าหลัก - Prediction Page

| ฟีเจอร์ | คำอธิบาย |
|--------|----------|
| **ช่องกรอก Headline** | รับพาดหัวข่าว |
| **ช่องกรอก Body** | รับเนื้อหาข่าว |
| **ปุ่ม Try Example** | สุ่มข่าวตัวอย่าง 5 ข่าว (Business, SciTech, World) มาเติมให้อัตโนมัติ |
| **ปุ่ม Predict** | ส่งข้อมูลไป Backend เพื่อทำนาย |
| **ปุ่ม Clear** | ล้างข้อมูลทั้งหมด |

### 2. ส่วนแสดงผล - Result Section

| ข้อมูล | คำอธิบาย |
|--------|----------|
| **Predicted Label** | หมวดหมู่ที่ทำนายได้ (Business, SciTech, World) |
| **Confidence Score** | ค่าความมั่นใจ (เช่น 95.5%) |
| **Probability Bar Chart** | กราฟแท่งแสดงความน่าจะเป็นของทุกคลาส |
| **Latency** | เวลาที่ใช้ประมวลผล (ms) |
| **Model Version** | เวอร์ชันของโมเดล |

### 3. หน้า Error Analysis

| ฟีเจอร์ | คำอธิบาย |
|--------|----------|
| **Static Gallery** | แสดงตัวอย่างที่โมเดลทำนายผิด 3 ตัวอย่าง |
| **Actual vs Predicted** | เปรียบเทียบ Label จริง vs ที่ทำนาย |
| **การวิเคราะห์สาเหตุ** | อธิบายสาเหตุของ Error (Mixed Signal, Domain Shift) |
| **ข้อเสนอแนะ** | แนวทางปรับปรุงโมเดล |

### 4. ฟีเจอร์เสริม

- ✅ **API Status Indicator** - แสดงสถานะ API แบบ Real-time (ออนไลน์/ออฟไลน์)
- ✅ **Model Info Panel** - แสดงข้อมูลโมเดล (Algorithm, Classes, Vocabulary Size, Version)
- ✅ **Responsive Design** - รองรับทั้ง Desktop และ Mobile
- ✅ **Loading Animation** - แสดงสถานะขณะรอผลลัพธ์

---

## ⚙️ System Workflow (การทำงานของระบบ)

```
┌─────────────────────────────────────────────────────────────────┐
│  1. User กรอกข่าว หรือกด Try Example                              │
│     ↓                                                            │
│  2. กด Predict → Frontend ส่ง POST /predict ไป Backend           │
│     ↓                                                            │
│  3. Backend รวม Headline + Body → Preprocessing → TF-IDF         │
│     ↓                                                            │
│  4. Logistic Regression ทำนาย → คืน Label + Probability          │
│     ↓                                                            │
│  5. Frontend แสดงผล: Label, Confidence, Bar Chart, Latency       │
└─────────────────────────────────────────────────────────────────┘
```

---

## �🚀 วิธีการรัน

### 1. ติดตั้ง Dependencies

```bash
cd frontend
npm install
```

### 2. ตั้งค่า Environment Variables

สร้างไฟล์ `.env` จาก `.env.example`:
```bash
cp .env.example .env
```

แก้ไข `.env` ตามต้องการ:
```env
VITE_API_URL=http://localhost:5001
```

### 3. รัน Development Server

```bash
npm run dev
```

เปิดเบราว์เซอร์ที่ http://localhost:5173

### 4. Build สำหรับ Production

```bash
npm run build
```

ไฟล์ build จะอยู่ที่ `dist/`

---

## ⚠️ สำคัญ

ต้องรัน Backend API ก่อน:
```bash
cd ../backend
source venv/bin/activate
gunicorn -w 4 -b 0.0.0.0:5001 app:app
```

---

## 🛠️ เทคโนโลยีที่ใช้

| Technology | Purpose |
|------------|---------|
| **Vite** | Build Tool & Dev Server |
| **React 18** | UI Framework |
| **Tailwind CSS** | Styling |
| **Lucide React** | Icons |
| **Fetch API** | HTTP Requests |

---

## 📁 โครงสร้างโปรเจค

```
frontend/
├── src/
│   ├── App.jsx       # Main Component (Prediction + Error Analysis)
│   ├── main.jsx      # React Entry Point
│   └── index.css     # Tailwind CSS Import
├── .env              # Environment Variables
├── .env.example      # Example ENV file
├── index.html        # HTML Template
├── package.json      # Dependencies
├── vite.config.js    # Vite Configuration (with Tailwind)
└── README.md         # This file
```

---

## 🔌 API Endpoints ที่เรียกใช้

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | ตรวจสอบสถานะ API |
| GET | `/model/info` | ดึงข้อมูลโมเดล |
| POST | `/predict` | ทำนายหมวดหมู่ข่าว |

---

## 📱 Screenshots

### หน้าหลัก (Prediction)
- กรอก Headline และ Body
- กด Try Example เพื่อสุ่มข่าวตัวอย่าง
- กด Predict เพื่อทำนาย
- ดูผลลัพธ์พร้อม Probability Bar Chart

### หน้า Error Analysis
- ดูตัวอย่างที่โมเดลทายผิด
- อ่านการวิเคราะห์สาเหตุ
- ดูข้อเสนอแนะในการปรับปรุง
