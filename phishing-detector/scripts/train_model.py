"""
Optimized training script for email classifier
Includes data cleaning and progress tracking
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


class EmailDataset(TorchDataset):
    """Custom PyTorch Dataset for email classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def clean_data(df, max_length=10000):
    """Clean and filter data"""
    print(f"\n🧹 Cleaning data...")
    print(f"   Original size: {len(df):,} samples")
    
    # Remove extremely long emails (likely corrupted)
    df = df[df['text'].str.len() < max_length].copy()
    print(f"   After length filter: {len(df):,} samples")
    
    # Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates(subset=['text']).copy()
    if len(df) < original_len:
        print(f"   Removed {original_len - len(df):,} duplicates")
    
    # Remove empty texts
    df = df[df['text'].str.strip().str.len() > 10].copy()
    
    print(f"   ✅ Final size: {len(df):,} samples\n")
    return df


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    batch_count = len(dataloader)
    print(f"   Training on {batch_count} batches...")
    
    for idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Get predictions
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        # Print progress every 500 batches
        if (idx + 1) % 500 == 0:
            print(f"      Batch {idx+1}/{batch_count} - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """Evaluate the model"""
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    batch_count = len(dataloader)
    print(f"   Evaluating on {batch_count} batches...")
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Print progress every 200 batches
            if (idx + 1) % 200 == 0:
                print(f"      Batch {idx+1}/{batch_count}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary'
    )
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_metrics(pred):
    """Compute metrics for evaluation"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_email_classifier(
    train_file: str,
    val_file: str,
    output_dir: str,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    quick_test: bool = False,
    max_text_length: int = 10000
):
    """
    Train email classifier with optimized approach
    
    Args:
        train_file: Path to training data CSV
        val_file: Path to validation data CSV
        output_dir: Output directory for model
        model_name: Pretrained model name
        num_epochs: Number of training epochs
        batch_size: Batch size
        quick_test: Use small subset for quick testing
        max_text_length: Maximum character length for emails
    """
    
    print("\n" + "="*70)
    print("🛡️  PHISHING EMAIL CLASSIFIER TRAINING")
    print("="*70 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Device: {device}")
    if device.type == 'cpu':
        print("   ⚠️  Training on CPU will be slower. GPU recommended for large datasets.\n")
    
    # Load data
    print(f"📖 Loading data...")
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    
    # Clean data
    train_df = clean_data(train_df, max_length=max_text_length)
    val_df = clean_data(val_df, max_length=max_text_length)
    
    # Use subset for quick testing
    if quick_test:
        print("⚡ QUICK TEST MODE: Using 1,000 train / 200 val samples")
        train_df = train_df.sample(n=min(1000, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(200, len(val_df)), random_state=42)
        print(f"   Quick train: {len(train_df):,} samples")
        print(f"   Quick val:   {len(val_df):,} samples\n")
    
    # Load tokenizer
    print(f"🤖 Loading tokenizer: {model_name}")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name, local_files_only=True)
        print("   ✅ Using cached tokenizer\n")
    except:
        print("   📥 Downloading tokenizer (first time only)...\n")
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    
    # Create datasets
    print("📦 Creating PyTorch datasets...")
    train_dataset = EmailDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer,
        max_length=512
    )
    val_dataset = EmailDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer,
        max_length=512
    )
    print(f"   ✅ Train dataset: {len(train_dataset):,} samples")
    print(f"   ✅ Val dataset:   {len(val_dataset):,} samples\n")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Load model
    print(f"🧠 Loading model: {model_name}")
    try:
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            local_files_only=True
        )
        print("   ✅ Using cached model\n")
    except:
        print("   📥 Downloading model (first time only, ~250MB)...\n")
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
    
    model.to(device)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("⚙️  TRAINING CONFIGURATION")
    print("="*70)
    print(f"   Epochs:          {num_epochs}")
    print(f"   Batch size:      {batch_size}")
    print(f"   Learning rate:   2e-5")
    print(f"   Max seq length:  512 tokens")
    print(f"   Training steps:  {total_steps}")
    print(f"   Output dir:      {output_dir}")
    print("="*70 + "\n")
    
    # Training loop
    print("🚀 STARTING TRAINING\n")
    
    best_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_metrics': []}
    
    for epoch in range(num_epochs):
        print(f"{'='*70}")
        print(f"📊 Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"   Train Loss:     {train_loss:.4f}")
        print(f"   Train Accuracy: {train_acc:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        print(f"\n   Val Loss:       {val_metrics['loss']:.4f}")
        print(f"   Val Accuracy:   {val_metrics['accuracy']:.4f}")
        print(f"   Val Precision:  {val_metrics['precision']:.4f}")
        print(f"   Val Recall:     {val_metrics['recall']:.4f}")
        print(f"   Val F1-Score:   {val_metrics['f1']:.4f}\n")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_metrics'].append(val_metrics)
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            print(f"   🎯 New best F1: {best_f1:.4f} - Saving model...")
            
            model_path = os.path.join(output_dir, 'best_model')
            os.makedirs(model_path, exist_ok=True)
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            print(f"   ✅ Saved to: {model_path}\n")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}')
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    
    print("="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📈 Final Results:")
    print(f"   Best F1-Score:  {best_f1:.4f}")
    print(f"   Final Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"   Final Precision:{val_metrics['precision']:.4f}")
    print(f"   Final Recall:   {val_metrics['recall']:.4f}")
    
    print(f"\n💾 Model saved to: {os.path.join(output_dir, 'best_model')}")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Test API:    python api/main.py")
    print(f"   2. Run Web UI:  streamlit run web/app.py")
    print(f"   3. Evaluate:    python scripts/evaluate_model.py\n")
    
    return model, tokenizer, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train phishing email classifier')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick test mode with 1,000 samples (~5 minutes)')
    parser.add_argument('--full', action='store_true',
                        help='Full training mode with all data (~30-60 minutes)')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    args = parser.parse_args()
    
    train_file = "data/processed/train.csv"
    val_file = "data/processed/val.csv"
    output_dir = "models/email_classifier"
    
    if not os.path.exists(train_file):
        print(f"❌ Error: Training data not found at {train_file}")
        print(f"   Please run: python scripts/download_kaggle_dataset.py")
        sys.exit(1)
    
    if not os.path.exists(val_file):
        print(f"❌ Error: Validation data not found at {val_file}")
        print(f"   Please run: python scripts/download_kaggle_dataset.py")
        sys.exit(1)
    
    # Default to quick mode unless --full is specified
    quick_mode = not args.full
    
    if quick_mode:
        print("\n" + "="*70)
        print("⚡ QUICK TEST MODE")
        print("   Training on 1,000 samples")
        print("   Estimated time: 5-10 minutes")
        print("   Use --full flag for complete training with all samples")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("🚀 FULL TRAINING MODE")
        print("   Training on ~75,000 samples")
        print("   Estimated time: 30-60 minutes")
        print("   You can interrupt with Ctrl+C and resume later")
        print("="*70)
    
    try:
        train_email_classifier(
            train_file, 
            val_file, 
            output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            quick_test=quick_mode
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("   Checkpoints saved to: models/email_classifier/")
        print("   You can resume training later")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
