# 🇹🇭 Thai News Topic Classifier - Frontend

Vite React Frontend สำหรับระบบจำแนกหมวดหมู่ข่าวภาษาไทย

## 🚀 วิธีการรัน

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

### 3. Build สำหรับ Production

```bash
npm run build
```

ไฟล์ build จะอยู่ที่ `dist/`

## ⚠️ สำคัญ

ต้องรัน Backend API ก่อน:
```bash
cd ../backend
python app.py
```

Backend จะรันที่ http://localhost:5000

## 🎨 Features

- ✅ กรอก Headline และ Body
- ✅ แสดง Label (หมวดหมู่ข่าว)
- ✅ แสดงค่า Confidence
- ✅ แสดง Probability แบบ Bar Chart
- ✅ แสดงข้อมูลโมเดล
- ✅ ตรวจสอบสถานะ API

## 📁 โครงสร้าง

```
frontend/
├── src/
│   ├── App.jsx     # Main Component
│   ├── App.css     # Styles
│   └── main.jsx    # Entry Point
├── index.html
├── package.json
└── vite.config.js
```
