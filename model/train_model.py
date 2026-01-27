#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai News Topic Classifier
===========================
โปรแกรมสำหรับจำแนกหมวดหมู่ข่าวภาษาไทย (Thai News Topic Classification)

ขั้นตอนการทำงาน:
1. Dataset Understanding - ทำความเข้าใจข้อมูล
2. Preprocessing - เตรียมข้อมูล
3. Baseline Model Training - สร้างโมเดล TF-IDF + Logistic Regression
4. Evaluation - ประเมินผล
5. Error Analysis - วิเคราะห์ข้อผิดพลาด

Author: Thai News Classifier Team
Date: 2026-01-27
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# ขั้นตอนที่ 1: Dataset Understanding (การทำความเข้าใจ Dataset)
# ============================================================================
def load_and_understand_dataset(csv_path: str) -> pd.DataFrame:
    """
    โหลดและวิเคราะห์ข้อมูลเบื้องต้น
    
    Dataset Description:
    - ประเภทข้อมูล: ข่าวภาษาไทย (Thai News Topic Dataset)
    - Input Features: headline (พาดหัว) + body (เนื้อหาข่าว)
    - Target Label: topic (เช่น SciTech, World, Business)
    - ลักษณะข้อมูล: train_easy version clean - ข้อมูลค่อนข้างสะอาด ความยากระดับง่าย
    
    Args:
        csv_path: พาธไปยังไฟล์ CSV
        
    Returns:
        DataFrame ที่โหลดมา
    """
    print("=" * 70)
    print("ขั้นตอนที่ 1: Dataset Understanding (การทำความเข้าใจ Dataset)")
    print("=" * 70)
    
    # โหลดข้อมูล
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 ข้อมูลทั่วไป:")
    print(f"   - จำนวนแถว (samples): {len(df):,}")
    print(f"   - จำนวนคอลัมน์: {len(df.columns)}")
    print(f"   - คอลัมน์: {list(df.columns)}")
    
    print(f"\n🏷️ การกระจายของ Target Label (topic):")
    topic_counts = df['topic'].value_counts()
    for topic, count in topic_counts.items():
        percentage = count / len(df) * 100
        print(f"   - {topic}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📝 ตัวอย่างข้อมูล (3 แถวแรก):")
    print(df[['headline', 'body', 'topic']].head(3).to_string())
    
    # ตรวจสอบ missing values
    print(f"\n⚠️ Missing Values:")
    missing = df[['headline', 'body', 'topic']].isnull().sum()
    for col, count in missing.items():
        print(f"   - {col}: {count}")
    
    # ตรวจสอบ version (clean/noisy)
    if 'version' in df.columns:
        print(f"\n🔍 Version Distribution:")
        version_counts = df['version'].value_counts()
        for version, count in version_counts.items():
            print(f"   - {version}: {count:,}")
    
    return df


# ============================================================================
# ขั้นตอนที่ 2: Preprocessing (การเตรียมข้อมูล)
# ============================================================================
def preprocess_text(text: str) -> str:
    """
    ทำ Preprocessing ข้อความ
    
    สิ่งที่ทำ:
    1. Whitespace Normalization - รวมช่องว่างหลายตัวเป็นตัวเดียว
    2. Strip - ตัดช่องว่างหัวท้าย
    3. Basic Normalization - แปลงตัวเลขไทยเป็นอารบิก (ถ้ามี)
    
    ข้อห้าม (ไม่ทำ over-cleaning):
    - ไม่ลบ emoji (อาจมีความหมายในบริบทข่าว)
    - ไม่ลบ slang (อาจเป็นส่วนหนึ่งของข่าว)
    - ไม่ลบตัวเลข (สำคัญสำหรับข่าว)
    - ไม่ลบเครื่องหมายวรรคตอน (สำคัญสำหรับความหมาย)
    
    เหตุผล:
    - ข้อมูลเป็น version clean อยู่แล้ว ไม่ต้อง clean มาก
    - การ over-cleaning อาจทำให้สูญเสียข้อมูลสำคัญ
    - TF-IDF จะจัดการกับ noise ได้ในระดับหนึ่ง
    
    Args:
        text: ข้อความต้นฉบับ
        
    Returns:
        ข้อความที่ผ่าน preprocessing
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # 1. Whitespace Normalization: รวมช่องว่างหลายตัวเป็นตัวเดียว
    #    เหตุผล: ช่องว่างซ้ำไม่มีความหมาย และอาจทำให้ TF-IDF นับคำผิด
    text = re.sub(r'\s+', ' ', text)
    
    # 2. Strip: ตัดช่องว่างหัวท้าย
    #    เหตุผล: ช่องว่างหัวท้ายไม่มีความหมาย
    text = text.strip()
    
    # 3. Thai Digits Normalization: แปลงตัวเลขไทยเป็นอารบิก
    #    เหตุผล: ให้ตัวเลขมีรูปแบบเดียวกัน
    thai_digits = '๐๑๒๓๔๕๖๗๘๙'
    arabic_digits = '0123456789'
    for thai, arabic in zip(thai_digits, arabic_digits):
        text = text.replace(thai, arabic)
    
    return text


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    เตรียม Features และ Labels
    
    Input Features: headline + " " + body
    Target Label: topic
    
    Args:
        df: DataFrame ต้นฉบับ
        
    Returns:
        tuple ของ (X, y) โดย X คือ text ที่รวมแล้ว, y คือ labels
    """
    print("\n" + "=" * 70)
    print("ขั้นตอนที่ 2: Preprocessing (การเตรียมข้อมูล)")
    print("=" * 70)
    
    print("\n📋 ขั้นตอนการ Preprocessing:")
    print("   1. Whitespace Normalization - รวมช่องว่างหลายตัวเป็นตัวเดียว")
    print("   2. Strip - ตัดช่องว่างหัวท้าย")
    print("   3. Thai Digits Normalization - แปลงตัวเลขไทยเป็นอารบิก")
    print("\n   ❌ สิ่งที่ไม่ทำ (ป้องกัน over-cleaning):")
    print("   - ไม่ลบ emoji (อาจมีความหมาย)")
    print("   - ไม่ลบ slang (อาจเป็นส่วนหนึ่งของข่าว)")
    print("   - ไม่ลบตัวเลขและเครื่องหมายวรรคตอน")
    
    # รวม headline + body
    df = df.copy()
    df['headline'] = df['headline'].fillna('')
    df['body'] = df['body'].fillna('')
    
    # สร้าง combined text
    df['text'] = df['headline'] + ' ' + df['body']
    
    # Apply preprocessing
    df['text'] = df['text'].apply(preprocess_text)
    
    X = df['text'].values
    y = df['topic'].values
    
    print(f"\n✅ เตรียมข้อมูลเสร็จสิ้น:")
    print(f"   - จำนวน samples: {len(X):,}")
    print(f"   - จำนวน classes: {len(np.unique(y))}")
    print(f"   - Classes: {list(np.unique(y))}")
    
    # แสดงตัวอย่าง text ที่รวมแล้ว
    print(f"\n📝 ตัวอย่าง text หลัง preprocessing (100 ตัวอักษรแรก):")
    for i in range(min(2, len(X))):
        sample = X[i][:100] + "..." if len(X[i]) > 100 else X[i]
        print(f"   [{i+1}] {sample}")
    
    return X, y, df


# ============================================================================
# ขั้นตอนที่ 3: Baseline Model Training
# ============================================================================
def train_baseline_model(X_train, y_train, X_test, y_test) -> tuple:
    """
    สร้างและ Train Baseline Model
    
    Baseline Model Configuration:
    1. Feature Extraction: TF-IDF (word-level)
       - ngram_range=(1, 2): ใช้ unigram และ bigram
       - max_features=10000: จำกัดคำศัพท์ไม่ให้ใหญ่เกินไป
       - sublinear_tf=True: ใช้ log scaling สำหรับ TF
       
    2. Model: Logistic Regression
       - class_weight='balanced': สำคัญ! จัดการกับ class imbalance
       - max_iter=1000: เพิ่ม iteration เพื่อให้ converge
       - solver='lbfgs': เหมาะกับ multiclass
       
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple ของ (vectorizer, model, y_pred)
    """
    print("\n" + "=" * 70)
    print("ขั้นตอนที่ 3: Baseline Model Training")
    print("=" * 70)
    
    # 1. TF-IDF Vectorizer
    print("\n🔧 1. สร้าง TF-IDF Vectorizer:")
    print("   - ngram_range: (1, 2) - ใช้ unigram และ bigram")
    print("   - max_features: 10,000")
    print("   - sublinear_tf: True - ใช้ log scaling")
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # word-level unigrams and bigrams
        max_features=10000,       # จำกัดขนาด vocabulary
        sublinear_tf=True,        # ใช้ 1 + log(tf) แทน tf
        min_df=2,                 # ละเว้นคำที่ปรากฏน้อยกว่า 2 ครั้ง
        max_df=0.95               # ละเว้นคำที่ปรากฏมากกว่า 95% ของ documents
    )
    
    print("   กำลัง fit vectorizer...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"   ✅ Vocabulary size: {len(vectorizer.vocabulary_):,}")
    print(f"   ✅ TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # 2. Logistic Regression
    print("\n🔧 2. สร้าง Logistic Regression Model:")
    print("   - class_weight: 'balanced' (สำคัญ!)")
    print("   - max_iter: 1000")
    print("   - solver: 'lbfgs'")
    print("   - multi_class: 'multinomial'")
    
    model = LogisticRegression(
        class_weight='balanced',  # สำคัญ! จัดการกับ class imbalance
        max_iter=1000,
        solver='lbfgs',
        random_state=42,
        n_jobs=-1
    )
    
    print("   กำลัง train model...")
    model.fit(X_train_tfidf, y_train)
    print("   ✅ Training เสร็จสิ้น!")
    
    # 3. Prediction
    y_pred = model.predict(X_test_tfidf)
    
    return vectorizer, model, y_pred


def save_model(vectorizer, model, output_dir: str):
    """
    บันทึกโมเดลเป็นไฟล์ .joblib
    
    Args:
        vectorizer: TF-IDF vectorizer
        model: Trained model
        output_dir: โฟลเดอร์สำหรับบันทึก
    """
    print("\n💾 บันทึกโมเดล:")
    
    # สร้างโฟลเดอร์ถ้ายังไม่มี
    os.makedirs(output_dir, exist_ok=True)
    
    # บันทึก vectorizer
    vectorizer_path = os.path.join(output_dir, 'tfidf_vectorizer.joblib')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"   ✅ Vectorizer saved: {vectorizer_path}")
    
    # บันทึก model
    model_path = os.path.join(output_dir, 'logistic_regression_model.joblib')
    joblib.dump(model, model_path)
    print(f"   ✅ Model saved: {model_path}")
    
    return vectorizer_path, model_path


# ============================================================================
# ขั้นตอนที่ 4: Evaluation (การประเมินผล)
# ============================================================================
def evaluate_model(y_test, y_pred, classes, output_dir: str = None) -> dict:
    """
    ประเมินผลโมเดล
    
    รายงาน:
    - Accuracy
    - Macro-F1
    - Confusion Matrix
    
    Args:
        y_test: Ground truth labels
        y_pred: Predicted labels
        classes: รายชื่อ classes
        output_dir: โฟลเดอร์สำหรับบันทึกรูป (optional)
        
    Returns:
        dict ของ metrics
    """
    print("\n" + "=" * 70)
    print("ขั้นตอนที่ 4: Evaluation (การประเมินผล)")
    print("=" * 70)
    
    # 1. Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 2. Macro-F1
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"📊 Macro-F1: {macro_f1:.4f}")
    
    # 3. Classification Report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("\n📊 Confusion Matrix:")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    
    # Plot confusion matrix
    if output_dir:
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes
        )
        plt.title('Confusion Matrix - Thai News Topic Classifier')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=150)
        print(f"\n💾 Confusion Matrix saved: {cm_path}")
        plt.close()
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'confusion_matrix': cm
    }


def get_misclassified_samples(df_test, y_test, y_pred, n_samples: int = 10) -> pd.DataFrame:
    """
    ดึงตัวอย่างที่โมเดลทำนายผิด
    
    Args:
        df_test: DataFrame ของ test set
        y_test: Ground truth labels
        y_pred: Predicted labels
        n_samples: จำนวนตัวอย่างที่ต้องการ
        
    Returns:
        DataFrame ของตัวอย่างที่ทำนายผิด
    """
    print(f"\n📝 ตัวอย่างที่ทำนายผิด (อย่างน้อย {n_samples} ตัวอย่าง):")
    print("-" * 70)
    
    # หา indices ที่ทำนายผิด
    wrong_mask = y_test != y_pred
    wrong_indices = np.where(wrong_mask)[0]
    
    print(f"   จำนวนที่ทำนายผิดทั้งหมด: {len(wrong_indices)} จาก {len(y_test)}")
    print(f"   อัตราผิดพลาด: {len(wrong_indices)/len(y_test)*100:.2f}%")
    
    # สุ่มเลือกตัวอย่าง
    sample_indices = wrong_indices[:min(n_samples, len(wrong_indices))]
    
    # สร้าง DataFrame
    misclassified = []
    for i, idx in enumerate(sample_indices):
        row = df_test.iloc[idx]
        headline = row['headline'][:50] + "..." if len(str(row['headline'])) > 50 else row['headline']
        body = row['body'][:100] + "..." if len(str(row['body'])) > 100 else row['body']
        
        misclassified.append({
            'index': idx,
            'headline': headline,
            'body': body,
            'actual': y_test[idx],
            'predicted': y_pred[idx]
        })
        
        print(f"\n   [{i+1}] Index: {idx}")
        print(f"       Headline: {headline}")
        print(f"       Body: {body}")
        print(f"       Actual: {y_test[idx]} → Predicted: {y_pred[idx]}")
    
    return pd.DataFrame(misclassified)


# ============================================================================
# ขั้นตอนที่ 5: Error Analysis (การวิเคราะห์ข้อผิดพลาด)
# ============================================================================
def analyze_errors(misclassified_df: pd.DataFrame, classes: list):
    """
    วิเคราะห์ข้อผิดพลาดและจัดกลุ่ม
    
    ประเภทข้อผิดพลาดที่พบบ่อย:
    1. Mixed Signal (ความกำกวมของภาษา)
       - ข่าวที่มีเนื้อหาทับซ้อนหลายหมวด
       - เช่น ข่าว Business ที่พูดถึง SciTech
       
    2. Domain Shift (คำศัพท์เฉพาะทาง)
       - คำศัพท์ที่โมเดลไม่เคยเห็นใน training
       - คำศัพท์ที่มีความหมายต่างในบริบทต่างกัน
       
    3. Typo/Noise (ปัญหาจากข้อมูล)
       - ตัวสะกดผิด
       - ข้อมูลที่ไม่สมบูรณ์
       
    Args:
        misclassified_df: DataFrame ของตัวอย่างที่ทำนายผิด
        classes: รายชื่อ classes
    """
    print("\n" + "=" * 70)
    print("ขั้นตอนที่ 5: Error Analysis (การวิเคราะห์ข้อผิดพลาด)")
    print("=" * 70)
    
    print("\n🔍 การจัดกลุ่มประเภทข้อผิดพลาด:")
    print("\n" + "-" * 60)
    
    # 1. Mixed Signal Analysis
    print("\n📌 ประเภทที่ 1: Mixed Signal (ความกำกวมของภาษา)")
    print("   คำอธิบาย: ข่าวที่มีเนื้อหาทับซ้อนหลายหมวดหมู่")
    print("   ตัวอย่าง: ข่าว Business ที่พูดถึงเทคโนโลยี อาจถูกทำนายเป็น SciTech")
    
    # นับ confusion pairs
    if len(misclassified_df) > 0:
        confusion_pairs = misclassified_df.groupby(['actual', 'predicted']).size()
        print("\n   คู่ที่สับสนบ่อย:")
        for (actual, predicted), count in confusion_pairs.head(5).items():
            print(f"   - {actual} → {predicted}: {count} ครั้ง")
    
    # 2. Domain Shift
    print("\n📌 ประเภทที่ 2: Domain Shift (คำศัพท์เฉพาะทาง)")
    print("   คำอธิบาย: คำศัพท์เฉพาะทางที่โมเดลไม่คุ้นเคย")
    print("   ตัวอย่าง: ")
    print("   - คำศัพท์ทางการเงินใหม่ๆ")
    print("   - ชื่อเทคโนโลยีล่าสุด")
    print("   - ศัพท์เฉพาะทางการทูต")
    
    # 3. Typo/Noise
    print("\n📌 ประเภทที่ 3: Typo/Noise (ปัญหาจากข้อมูล)")
    print("   คำอธิบาย: ข้อมูลที่มีความผิดปกติ")
    print("   ตัวอย่าง:")
    print("   - ตัวสะกดผิด")
    print("   - headline หรือ body สั้นเกินไป")
    print("   - ข้อความที่ไม่สมบูรณ์")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("💡 แนวทางแก้ไข (Recommendations):")
    print("=" * 60)
    
    print("\n1. สำหรับ Mixed Signal:")
    print("   - เพิ่ม features จาก subtopic เพื่อช่วยแยกแยะ")
    print("   - ใช้ hierarchical classification")
    print("   - ลองใช้ multi-label classification")
    
    print("\n2. สำหรับ Domain Shift:")
    print("   - เพิ่มข้อมูล training ที่หลากหลาย")
    print("   - ใช้ word embeddings แทน TF-IDF")
    print("   - ใช้ pre-trained Thai language models (เช่น WangchanBERTa)")
    
    print("\n3. สำหรับ Typo/Noise:")
    print("   - เพิ่มขั้นตอน spell checking")
    print("   - ใช้ character-level features เพิ่มเติม")
    print("   - กรองข้อมูลที่สั้นเกินไปออก")
    
    print("\n💡 แนวทางแก้ไขหลัก (เสนอ 1 แนวทาง):")
    print("-" * 60)
    print("""
    🎯 ใช้ Pre-trained Thai Language Model (WangchanBERTa)
    
    เหตุผล:
    1. สามารถเข้าใจความหมายเชิงบริบท (contextual understanding)
    2. จัดการกับ Mixed Signal ได้ดีกว่า TF-IDF
    3. ทนต่อ typo และ noise ได้ดีกว่า
    4. ไม่ต้องทำ feature engineering มาก
    
    ขั้นตอนการ implement:
    - pip install transformers pythainlp
    - ใช้ model: airesearch/wangchanberta-base-att-spm-uncased
    - Fine-tune บน dataset นี้
    """)


# ============================================================================
# Main Function
# ============================================================================
def main():
    """
    Main function สำหรับ train Thai News Topic Classifier
    """
    print("=" * 70)
    print("🇹🇭 Thai News Topic Classifier")
    print("=" * 70)
    
    # Configuration
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(SCRIPT_DIR, '12.agnews_thai_train_easy.csv')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    print(f"\n⚙️ Configuration:")
    print(f"   - CSV Path: {CSV_PATH}")
    print(f"   - Output Dir: {OUTPUT_DIR}")
    print(f"   - Test Size: {TEST_SIZE}")
    print(f"   - Random State: {RANDOM_STATE}")
    
    # ขั้นตอนที่ 1: Load and understand dataset
    df = load_and_understand_dataset(CSV_PATH)
    
    # ขั้นตอนที่ 2: Preprocess data
    X, y, df_processed = prepare_features(df)
    
    # แบ่ง Train/Test
    print("\n" + "=" * 70)
    print("การแบ่งข้อมูล Train/Test")
    print("=" * 70)
    
    # สร้าง indices สำหรับ split
    indices = np.arange(len(X))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # รักษาสัดส่วน class
    )
    
    print(f"\n📊 การแบ่งข้อมูล:")
    print(f"   - Training set: {len(X_train):,} samples ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"   - Test set: {len(X_test):,} samples ({TEST_SIZE*100:.0f}%)")
    
    # สร้าง df_test สำหรับ error analysis
    df_test = df_processed.iloc[idx_test].reset_index(drop=True)
    
    # ขั้นตอนที่ 3: Train baseline model
    vectorizer, model, y_pred = train_baseline_model(X_train, y_train, X_test, y_test)
    
    # บันทึกโมเดล
    save_model(vectorizer, model, OUTPUT_DIR)
    
    # ขั้นตอนที่ 4: Evaluate model
    classes = sorted(df['topic'].unique())
    metrics = evaluate_model(y_test, y_pred, classes, OUTPUT_DIR)
    
    # ดึงตัวอย่างที่ทำนายผิด
    misclassified_df = get_misclassified_samples(df_test, y_test, y_pred, n_samples=10)
    
    # บันทึก misclassified samples
    if len(misclassified_df) > 0:
        misclassified_path = os.path.join(OUTPUT_DIR, 'misclassified_samples.csv')
        misclassified_df.to_csv(misclassified_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 Misclassified samples saved: {misclassified_path}")
    
    # ขั้นตอนที่ 5: Error Analysis
    analyze_errors(misclassified_df, classes)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print(f"\n   ✅ Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   ✅ Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"\n   📁 Output files saved to: {OUTPUT_DIR}")
    print(f"      - tfidf_vectorizer.joblib")
    print(f"      - logistic_regression_model.joblib")
    print(f"      - confusion_matrix.png")
    print(f"      - misclassified_samples.csv")
    
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
