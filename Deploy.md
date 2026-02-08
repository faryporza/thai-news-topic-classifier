# 🚀 Deployment Guide

คู่มือการ Deploy แอปพลิเคชัน Thai News Topic Classifier

---

## 📋 สารบัญ

1. [Backend - Google Cloud Run](#backend---google-cloud-run)
2. [Frontend - Vercel](#frontend---vercel)
3. [การเชื่อมต่อ Backend และ Frontend](#การเชื่อมต่อ-backend-และ-frontend)

---

## Backend - Google Cloud Run

### ข้อกำหนดเบื้องต้น

- ติดตั้ง [Google Cloud CLI (gcloud)](https://cloud.google.com/sdk/docs/install)
- มีบัญชี Google Cloud และสร้าง Project แล้ว
- เปิดใช้งาน billing สำหรับ Project

### ขั้นตอนที่ 1: ตั้งค่า Google Cloud CLI

```bash
# Login เข้า Google Cloud
gcloud auth login

# ตั้งค่า Project (แทนที่ YOUR_PROJECT_ID ด้วย Project ID ของคุณ)
gcloud config set project YOUR_PROJECT_ID

# เปิดใช้งาน Cloud Run API และ Artifact Registry
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

### ขั้นตอนที่ 2: สร้าง Artifact Registry Repository

```bash
# สร้าง Docker repository
gcloud artifacts repositories create thai-news-classifier \
  --repository-format=docker \
  --location=asia-southeast1 \
  --description="Thai News Topic Classifier Images"

# ตั้งค่า Docker authentication
gcloud auth configure-docker asia-southeast1-docker.pkg.dev
```

### ขั้นตอนที่ 3: Build และ Push Docker Image

```bash
# เข้าไปใน backend directory
cd backend

# Build Docker image
docker build -t asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/thai-news-classifier/backend:latest .

# Push image ไปยัง Artifact Registry
docker push asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/thai-news-classifier/backend:latest
```

### ขั้นตอนที่ 4: Deploy ไปยัง Cloud Run

```bash
gcloud run deploy thai-news-classifier-api \
  --image asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/thai-news-classifier/backend:latest \
  --platform managed \
  --region asia-southeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 120 \
  --min-instances 0 \
  --max-instances 10 \
  --port 8080
```

### ขั้นตอนที่ 5: ดู URL ของ Service

เมื่อ deploy สำเร็จ จะได้ URL ในรูปแบบ:
```
https://thai-news-classifier-api-xxxxxxxx-as.a.run.app
```

**เก็บ URL นี้ไว้สำหรับใช้ตั้งค่า Frontend**

### 🔄 การ Update Deployment

```bash
# Build และ Push image ใหม่
cd backend
docker build -t asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/thai-news-classifier/backend:latest .
docker push asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/thai-news-classifier/backend:latest

# Deploy version ใหม่
gcloud run deploy thai-news-classifier-api \
  --image asia-southeast1-docker.pkg.dev/YOUR_PROJECT_ID/thai-news-classifier/backend:latest \
  --region asia-southeast1
```

### 📊 การตรวจสอบ Logs

```bash
# ดู logs แบบ real-time
gcloud run logs tail thai-news-classifier-api --region asia-southeast1

# ดู logs ย้อนหลัง
gcloud run logs read thai-news-classifier-api --region asia-southeast1 --limit 100
```

---

## Frontend - Vercel

### ข้อกำหนดเบื้องต้น

- มีบัญชี [Vercel](https://vercel.com)
- ติดตั้ง [Vercel CLI](https://vercel.com/docs/cli) (optional)
- Push โค้ดไปยัง GitHub/GitLab/Bitbucket

### วิธีที่ 1: Deploy ผ่าน Vercel Dashboard (แนะนำ)

#### ขั้นตอนที่ 1: Import Project

1. ไปที่ [vercel.com/new](https://vercel.com/new)
2. เชื่อมต่อกับ GitHub repository ของคุณ
3. เลือก repository `thai-news-topic-classifier`
4. **สำคัญ:** ตั้งค่า Root Directory เป็น `frontend`

#### ขั้นตอนที่ 2: ตั้งค่า Build Settings

| Setting | Value |
|---------|-------|
| Framework Preset | Vite |
| Root Directory | `frontend` |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Install Command | `npm install` |

#### ขั้นตอนที่ 3: ตั้งค่า Environment Variables

เพิ่ม Environment Variables ใน Vercel Dashboard:

| Variable | Value | Description |
|----------|-------|-------------|
| `VITE_API_URL` | `https://thai-news-classifier-api-xxxxxxxx-as.a.run.app` | URL ของ Backend API (Cloud Run) |
| `VITE_AZURE_OPENAI_ENDPOINT` | `https://your-resource.cognitiveservices.azure.com/openai/responses` | Azure OpenAI Endpoint (ถ้าใช้) |
| `VITE_AZURE_OPENAI_API_KEY` | `your_api_key` | Azure OpenAI API Key (ถ้าใช้) |
| `VITE_AZURE_OPENAI_API_VERSION` | `2025-04-01-preview` | Azure OpenAI API Version |

#### ขั้นตอนที่ 4: Deploy

คลิก **Deploy** และรอจนกว่าจะ deploy เสร็จสมบูรณ์

---

### วิธีที่ 2: Deploy ผ่าน Vercel CLI

```bash
# ติดตั้ง Vercel CLI
npm install -g vercel

# Login
vercel login

# เข้าไปใน frontend directory
cd frontend

# Deploy (ครั้งแรก)
vercel

# Deploy to Production
vercel --prod
```

> **หมายเหตุ:** เมื่อใช้ CLI ครั้งแรก จะถูกถามให้ตั้งค่า:
> - Set up and deploy? → Yes
> - Which scope? → เลือก team ของคุณ
> - Link to existing project? → No (ถ้าสร้างใหม่)
> - Project name → thai-news-classifier
> - Directory → ./

### 🔄 การ Update Deployment

เมื่อ push โค้ดไปยัง repository, Vercel จะ auto-deploy ให้อัตโนมัติ

หรือสามารถ deploy แบบ manual ได้:
```bash
cd frontend
vercel --prod
```

---

## การเชื่อมต่อ Backend และ Frontend

### 1. ตรวจสอบ CORS บน Backend

ไฟล์ `backend/app.py` ควรมีการตั้งค่า CORS ที่อนุญาต Vercel domain:

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:5173",
    "https://your-app.vercel.app",
    "https://*.vercel.app"
])
```

### 2. ตั้งค่า Environment Variable บน Vercel

ตรวจสอบว่า `VITE_API_URL` ชี้ไปยัง Cloud Run URL ที่ถูกต้อง:

```
VITE_API_URL=https://thai-news-classifier-api-xxxxxxxx-as.a.run.app
```

### 3. ทดสอบการเชื่อมต่อ

เปิด Browser DevTools (F12) → Network tab และทดสอบ classify ข่าว เพื่อตรวจสอบว่า request ไปถึง Backend ได้สำเร็จ

---

## 🔒 Security Best Practices

### Backend (Cloud Run)
- ใช้ Secret Manager สำหรับ API Keys
- ตั้งค่า CORS ให้รัดกุม (ระบุ domains ที่อนุญาตเท่านั้น)
- ใช้ HTTPS เท่านั้น (Cloud Run บังคับใช้อยู่แล้ว)

### Frontend (Vercel)
- ใช้ Environment Variables ใน Vercel Dashboard (ไม่ commit `.env` ไปกับ code)
- ตรวจสอบว่าไม่มี secrets ใน client-side code

---

## 💰 ค่าใช้จ่ายโดยประมาณ

### Google Cloud Run
- **Free tier:** 2 ล้าน requests/เดือน, 360,000 GB-seconds
- หลังจาก free tier: ~$0.00002400/request

### Vercel
- **Hobby (Free):** เหมาะสำหรับ personal projects
- **Pro:** $20/เดือน สำหรับ commercial projects

---

## 🆘 Troubleshooting

### Backend ไม่ response
```bash
# ตรวจสอบ status
gcloud run services describe thai-news-classifier-api --region asia-southeast1

# ตรวจสอบ logs
gcloud run logs read thai-news-classifier-api --region asia-southeast1
```

### Frontend ไม่ได้รับข้อมูลจาก API
1. ตรวจสอบ CORS settings บน Backend
2. ตรวจสอบ `VITE_API_URL` ใน Vercel Environment Variables
3. ดู Browser Console สำหรับ error messages

### Build failed บน Vercel
1. ตรวจสอบ Root Directory = `frontend`
2. ตรวจสอบ Node.js version ใน package.json
3. ดู Build Logs ใน Vercel Dashboard

---

## 📝 Checklist ก่อน Deploy

- [ ] ทดสอบ build บน local (`docker build .` และ `npm run build`)
- [ ] ตรวจสอบ environment variables
- [ ] ตั้งค่า CORS บน Backend
- [ ] ทดสอบ API endpoints
- [ ] Push code ไปยัง repository

