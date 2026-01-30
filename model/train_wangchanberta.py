#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thai News Topic Classifier - WangchanBERTa Edition
===================================================
โปรแกรมสำหรับจำแนกหมวดหมู่ข่าวภาษาไทยด้วย WangchanBERTa

ทำไมถึงเปลี่ยนจาก TF-IDF + Logistic Regression มาใช้ WangchanBERTa?
---------------------------------------------------------------------
1. Contextual Understanding: เข้าใจความหมายเชิงบริบท
   - คำว่า "ตลาด" ในข่าวหุ้น vs ข่าวต่างประเทศ → BERT แยกได้
   - TF-IDF มองทุกคำเป็นอิสระ ไม่เข้าใจบริบท

2. Mixed Signal Handling: จัดการข่าวที่มีหลายสัญญาณได้ดี
   - ข่าว Business ที่พูดถึง AI → TF-IDF อาจสับสน
   - WangchanBERTa เข้าใจว่าประเด็นหลักคืออะไร

3. Robust to Noise: ทนต่อ typo และ noise
   - ตัวสะกดผิดเล็กน้อย → BERT ยังเข้าใจ
   - TF-IDF จะไม่รู้จักคำที่สะกดผิด

4. Transfer Learning: ใช้ความรู้จากข้อความภาษาไทยจำนวนมาก
   - Pre-trained บน corpus ภาษาไทยขนาดใหญ่
   - ไม่ต้องเริ่มเรียนรู้จาก 0

Expected Performance:
- TF-IDF + LR: ~85-90% accuracy
- WangchanBERTa: ~92-97% accuracy

Author: Thai News Classifier Team
Date: 2026-01-29
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================
class Config:
    """การตั้งค่าสำหรับ Training"""
    
    # Model Configuration
    MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
    NUM_LABELS = 4  # Business, SciTech, World, Sports (ถ้ามี) หรือปรับตาม dataset
    MAX_LENGTH = 256  # ความยาวสูงสุดของ input (tokens)
    
    # Training Configuration
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Data Split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(SCRIPT_DIR, "12.agnews_thai_train_easy.csv")
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_bert")
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "wangchanberta_model")


# ============================================================================
# Dataset Class
# ============================================================================
class ThaiNewsDataset(Dataset):
    """Dataset class สำหรับ Thai News"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================
def load_and_preprocess_data(csv_path: str) -> tuple:
    """
    โหลดและเตรียมข้อมูล
    
    Returns:
        tuple: (texts, labels, label2id, id2label)
    """
    print("=" * 70)
    print("📊 ขั้นตอนที่ 1: Loading & Preprocessing Data")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n✅ Loaded {len(df):,} samples")
    
    # Show class distribution
    print(f"\n📋 Class Distribution:")
    for topic, count in df['topic'].value_counts().items():
        pct = count / len(df) * 100
        print(f"   - {topic}: {count:,} ({pct:.1f}%)")
    
    # Combine headline + body
    df['headline'] = df['headline'].fillna('')
    df['body'] = df['body'].fillna('')
    df['text'] = df['headline'] + ' ' + df['body']
    
    # Basic preprocessing
    def preprocess(text):
        text = str(text)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        return text
    
    df['text'] = df['text'].apply(preprocess)
    
    # Create label mappings
    labels_unique = sorted(df['topic'].unique())
    label2id = {label: idx for idx, label in enumerate(labels_unique)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"\n🏷️ Label Mapping:")
    for label, idx in label2id.items():
        print(f"   {label} → {idx}")
    
    # Convert labels to integers
    df['label_id'] = df['topic'].map(label2id)
    
    texts = df['text'].values
    labels = df['label_id'].values
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   - Text samples: {len(texts):,}")
    print(f"   - Unique labels: {len(label2id)}")
    
    return texts, labels, label2id, id2label


# ============================================================================
# Model Training
# ============================================================================
def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }


def train_wangchanberta(
    X_train, y_train, X_val, y_val,
    label2id, id2label, config
):
    """
    Train WangchanBERTa model
    """
    print("\n" + "=" * 70)
    print("🤖 ขั้นตอนที่ 2: Training WangchanBERTa")
    print("=" * 70)
    
    print(f"\n⚙️ Configuration:")
    print(f"   - Model: {config.MODEL_NAME}")
    print(f"   - Max Length: {config.MAX_LENGTH}")
    print(f"   - Batch Size: {config.BATCH_SIZE}")
    print(f"   - Learning Rate: {config.LEARNING_RATE}")
    print(f"   - Epochs: {config.NUM_EPOCHS}")
    
    # Load tokenizer
    print(f"\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create datasets
    print(f"📦 Creating datasets...")
    train_dataset = ThaiNewsDataset(X_train, y_train, tokenizer, config.MAX_LENGTH)
    val_dataset = ThaiNewsDataset(X_val, y_val, tokenizer, config.MAX_LENGTH)
    
    print(f"   - Train samples: {len(train_dataset):,}")
    print(f"   - Validation samples: {len(val_dataset):,}")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n🔧 Device: {device}")
    
    # Load model
    print(f"\n📥 Loading WangchanBERTa model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Create output directory
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE * 2,
        learning_rate=config.LEARNING_RATE,
        warmup_ratio=config.WARMUP_RATIO,
        weight_decay=config.WEIGHT_DECAY,
        logging_dir=os.path.join(config.OUTPUT_DIR, 'logs'),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    # Train
    print(f"\n🚀 Starting training...")
    print("-" * 50)
    train_result = trainer.train()
    
    print(f"\n✅ Training complete!")
    print(f"   - Training time: {train_result.metrics['train_runtime']:.2f} seconds")
    print(f"   - Training loss: {train_result.metrics['train_loss']:.4f}")
    
    # Save model
    print(f"\n💾 Saving model to {config.MODEL_SAVE_PATH}")
    trainer.save_model(config.MODEL_SAVE_PATH)
    tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
    
    return trainer, model, tokenizer


# ============================================================================
# Evaluation
# ============================================================================
def evaluate_model(trainer, X_test, y_test, id2label, config):
    """
    Evaluate the trained model
    """
    print("\n" + "=" * 70)
    print("📊 ขั้นตอนที่ 3: Evaluation")
    print("=" * 70)
    
    # Get predictions
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_PATH)
    test_dataset = ThaiNewsDataset(X_test, y_test, tokenizer, config.MAX_LENGTH)
    
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n📈 Performance Metrics:")
    print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   ✅ Macro-F1: {macro_f1:.4f}")
    
    # Classification Report
    labels = list(id2label.values())
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📊 Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title('Confusion Matrix - WangchanBERTa Thai News Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix_bert.png')
    plt.savefig(cm_path, dpi=150)
    print(f"\n💾 Confusion matrix saved: {cm_path}")
    plt.close()
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'predictions': y_pred
    }


# ============================================================================
# Model Comparison Summary
# ============================================================================
def print_comparison_summary():
    """Print comparison between TF-IDF and WangchanBERTa"""
    print("\n" + "=" * 70)
    print("📊 Model Comparison: TF-IDF vs WangchanBERTa")
    print("=" * 70)
    
    comparison = """
    ┌─────────────────────┬──────────────────────┬──────────────────────┐
    │ Aspect              │ TF-IDF + LR          │ WangchanBERTa        │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Accuracy (expected) │ ~85-90%              │ ~92-97%              │
    │ Context Understanding│ ❌ No               │ ✅ Yes               │
    │ Mixed Signal        │ ❌ Struggles         │ ✅ Handles well      │
    │ Typo Tolerance      │ ❌ Low               │ ✅ High              │
    │ Training Speed      │ ✅ Fast (seconds)    │ ❌ Slow (minutes)    │
    │ Inference Speed     │ ✅ Very Fast         │ ⚠️ Moderate          │
    │ Model Size          │ ✅ Small (~10 MB)    │ ❌ Large (~400 MB)   │
    │ GPU Required        │ ❌ No                │ ⚠️ Recommended       │
    └─────────────────────┴──────────────────────┴──────────────────────┘
    
    🎯 Recommendation:
    - Production with high accuracy requirement → WangchanBERTa
    - Quick prototyping / low resource → TF-IDF + Logistic Regression
    """
    print(comparison)


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main training function"""
    print("=" * 70)
    print("🇹🇭 Thai News Topic Classifier - WangchanBERTa Edition")
    print("=" * 70)
    print(f"\n📅 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config = Config()
    
    # Step 1: Load data
    texts, labels, label2id, id2label = load_and_preprocess_data(config.CSV_PATH)
    
    # Update config with actual number of labels
    config.NUM_LABELS = len(label2id)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels
    )
    
    print(f"\n📊 Data Split:")
    print(f"   - Training: {len(X_train):,} samples")
    print(f"   - Validation: {len(X_val):,} samples")
    
    # Step 2: Train model
    trainer, model, tokenizer = train_wangchanberta(
        X_train, y_train, X_val, y_val,
        label2id, id2label, config
    )
    
    # Step 3: Evaluate
    metrics = evaluate_model(trainer, X_val, y_val, id2label, config)
    
    # Print comparison
    print_comparison_summary()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 Training Complete!")
    print("=" * 70)
    print(f"\n📊 Final Results:")
    print(f"   ✅ Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   ✅ Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"\n📁 Model saved to: {config.MODEL_SAVE_PATH}")
    print(f"\n📅 Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
