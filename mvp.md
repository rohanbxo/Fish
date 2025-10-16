# NLP-Based Phishing Email and URL Detector - MVP Documentation

## Executive Summary

This document provides a complete guide to building a Minimum Viable Product (MVP) for an AI-powered phishing detection system. The MVP will demonstrate core capabilities in detecting phishing emails and malicious URLs using NLP and machine learning techniques, deployable as a REST API with a basic web interface.

**MVP Timeline:** 4-6 weeks for a functional prototype **Core Value:** Automated detection of sophisticated phishing attempts that bypass traditional keyword-based filters

---

## 1. Project Scope and MVP Features

### 1.1 In-Scope Features (MVP)

**Core Detection Capabilities:**

- Email text classification (phishing vs legitimate)
- URL analysis and risk scoring
- Combined email + URL assessment
- REST API for programmatic access
- Simple web interface for testing

**Technical Components:**

- Fine-tuned transformer model (DistilBERT or RoBERTa base)
- URL feature extraction and classification
- Hybrid scoring system combining text and URL analysis
- Basic confidence scoring and explanations

### 1.2 Out-of-Scope (Post-MVP)

- Real-time email client integration
- Browser extension
- Federated learning implementation
- Advanced sender reputation systems
- Multi-language support
- Automated phishing variant generation
- Large-scale deployment infrastructure

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────┐
│   Web Interface │
│   (Streamlit)   │
└────────┬────────┘
         │
    ┌────▼─────┐
    │ REST API │
    │ (FastAPI)│
    └────┬─────┘
         │
    ┌────▼──────────────────────┐
    │   Detection Pipeline      │
    ├───────────────────────────┤
    │ • Email Preprocessor      │
    │ • NLP Model (DistilBERT)  │
    │ • URL Feature Extractor   │
    │ • URL Classifier          │
    │ • Scoring Engine          │
    └───────────────────────────┘
         │
    ┌────▼─────┐
    │  Models  │
    │  & Data  │
    └──────────┘
```

### 2.2 Technology Stack

|Component|Technology|Rationale|
|---|---|---|
|NLP Model|DistilBERT|Faster than BERT, 97% performance, smaller footprint|
|Framework|PyTorch + HuggingFace|Industry standard, excellent pretrained models|
|API|FastAPI|High performance, async support, automatic docs|
|Web UI|Streamlit|Rapid prototyping, Python-native|
|Data Processing|Pandas, scikit-learn|Standard data science tools|
|URL Analysis|BeautifulSoup, tldextract|URL parsing and feature extraction|
|Storage|JSON/Pickle|Simple persistence for MVP|

---

## 3. Implementation Roadmap

### Phase 1: Data Preparation (Week 1)

**Objectives:**

- Acquire and prepare datasets
- Perform exploratory data analysis
- Create train/validation/test splits

**Tasks:**

1. **Dataset Acquisition**
    - Download Kaggle Phishing Email dataset
    - Obtain PhishTank URL data
    - Collect 2,000-5,000 samples for MVP (balanced classes)
2. **Data Preprocessing**

python

```python
   # Key preprocessing steps
   - Remove HTML tags and extract plain text
   - Normalize whitespace and special characters
   - Extract email headers (From, Subject, Reply-To)
   - Parse and extract URLs from email body
   - Label verification and cleaning
```

3. **Exploratory Analysis**
    - Class distribution analysis
    - Text length statistics
    - Common phishing indicators (urgency words, suspicious domains)
    - URL feature distributions

**Deliverables:**

- Clean, labeled dataset (CSV format)
- EDA notebook with visualizations
- Data statistics report

---

### Phase 2: Model Development - Email Classification (Week 2)

**Objectives:**

- Fine-tune transformer model for email classification
- Establish baseline performance metrics
- Implement evaluation framework

**Implementation Steps:**

1. **Baseline Model**

python

```python
   # Simple TF-IDF + Logistic Regression baseline
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.linear_model import LogisticRegression
   
   # Establish minimum acceptable performance
   # Target: 85%+ accuracy as baseline
```

2. **Transformer Fine-Tuning**

python

```python
   from transformers import (
       DistilBertTokenizer, 
       DistilBertForSequenceClassification,
       Trainer, 
       TrainingArguments
   )
   
   # Configuration
   model_name = "distilbert-base-uncased"
   num_labels = 2  # phishing vs legitimate
   max_length = 512  # token limit
   
   # Training parameters
   batch_size = 16
   learning_rate = 2e-5
   epochs = 3-5
```

3. **Feature Engineering**
    - Email header features (sender domain, reply-to mismatch)
    - Text features (urgency indicators, personal information requests)
    - Structural features (HTML complexity, link count)
4. **Model Evaluation**
    - Accuracy, Precision, Recall, F1-score
    - Confusion matrix analysis
    - ROC-AUC curve
    - False positive/negative analysis

**Deliverables:**

- Fine-tuned DistilBERT model (saved checkpoint)
- Training notebook with metrics
- Model evaluation report
- Baseline comparison results

**Target Metrics:**

- Accuracy: >92%
- Precision: >90% (minimize false alarms)
- Recall: >88% (catch most phishing)
- F1-Score: >90%

---

### Phase 3: URL Analysis Module (Week 2-3)

**Objectives:**

- Build URL feature extraction pipeline
- Train URL classifier
- Integrate with email detection

**URL Features to Extract:**

1. **Lexical Features**
    - URL length
    - Number of special characters (@, -, _, etc.)
    - Number of subdomains
    - Presence of IP address
    - Use of HTTPS
    - URL entropy (randomness measure)
2. **Domain Features**
    - Domain length
    - Top-level domain (TLD)
    - Age of domain (if WHOIS available)
    - Suspicious TLDs (.tk, .ml, .ga)
3. **Content Features**
    - Suspicious keywords in path
    - Shortened URL indicators
    - Redirection count

**Implementation:**

python

```python
import tldextract
import re
from urllib.parse import urlparse
import math

class URLFeatureExtractor:
    def extract_features(self, url):
        features = {}
        
        # Parse URL
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        
        # Lexical features
        features['length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['has_ip'] = self._has_ip_address(url)
        features['is_https'] = 1 if parsed.scheme == 'https' else 0
        
        # Domain features
        features['domain_length'] = len(ext.domain)
        features['subdomain_count'] = len(ext.subdomain.split('.'))
        features['suspicious_tld'] = self._check_suspicious_tld(ext.suffix)
        
        # Entropy
        features['entropy'] = self._calculate_entropy(url)
        
        return features
    
    def _calculate_entropy(self, string):
        prob = [float(string.count(c)) / len(string) 
                for c in dict.fromkeys(list(string))]
        entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy
```

**URL Classifier:**

- Random Forest or Gradient Boosting (XGBoost)
- Train on extracted features
- Target: >85% accuracy on URL classification

**Deliverables:**

- URL feature extractor module
- Trained URL classifier
- Feature importance analysis
- URL detection evaluation report

---

### Phase 4: Integration and Scoring System (Week 3-4)

**Objectives:**

- Combine email and URL detection
- Implement risk scoring algorithm
- Create explanation generation

**Hybrid Scoring System:**

python

```python
class PhishingDetector:
    def __init__(self, email_model, url_classifier):
        self.email_model = email_model
        self.url_classifier = url_classifier
        
    def detect(self, email_text, urls):
        # Email analysis
        email_score = self.email_model.predict_proba(email_text)[0][1]
        
        # URL analysis
        url_scores = []
        if urls:
            for url in urls:
                features = self.extract_url_features(url)
                url_score = self.url_classifier.predict_proba(features)[0][1]
                url_scores.append(url_score)
        
        # Combined scoring (weighted average)
        if url_scores:
            max_url_score = max(url_scores)
            final_score = 0.6 * email_score + 0.4 * max_url_score
        else:
            final_score = email_score
        
        # Risk categorization
        risk_level = self._categorize_risk(final_score)
        
        # Generate explanation
        explanation = self._generate_explanation(
            email_score, url_scores, final_score
        )
        
        return {
            'risk_score': final_score,
            'risk_level': risk_level,
            'email_score': email_score,
            'url_scores': url_scores,
            'explanation': explanation
        }
    
    def _categorize_risk(self, score):
        if score >= 0.8:
            return "HIGH - Likely Phishing"
        elif score >= 0.5:
            return "MEDIUM - Suspicious"
        elif score >= 0.3:
            return "LOW - Potentially Suspicious"
        else:
            return "SAFE - Likely Legitimate"
```

**Explanation Generation:**

Provide interpretable results:

- Key indicators found (urgency language, suspicious links)
- Specific features that triggered high scores
- Confidence level
- Actionable recommendations

**Deliverables:**

- Integrated detection pipeline
- Scoring algorithm implementation
- Explanation generator
- Integration testing suite

---

### Phase 5: API Development (Week 4)

**Objectives:**

- Build REST API for detection service
- Implement request/response handling
- Add basic security and validation

**API Endpoints:**

python

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import List, Optional

app = FastAPI(title="Phishing Detection API")

class EmailRequest(BaseModel):
    subject: str
    body: str
    sender: EmailStr
    urls: Optional[List[str]] = []

class DetectionResponse(BaseModel):
    risk_score: float
    risk_level: str
    email_score: float
    url_scores: List[float]
    explanation: dict
    indicators: List[str]

@app.post("/api/v1/detect", response_model=DetectionResponse)
async def detect_phishing(request: EmailRequest):
    """
    Analyze email and URLs for phishing indicators
    """
    try:
        # Combine subject and body
        email_text = f"{request.subject}\n\n{request.body}"
        
        # Run detection
        result = detector.detect(email_text, request.urls)
        
        return DetectionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.get("/api/v1/stats")
async def get_stats():
    """Return model performance statistics"""
    return {
        "accuracy": 0.93,
        "precision": 0.91,
        "recall": 0.89,
        "f1_score": 0.90
    }
```

**API Features:**

- Request validation
- Error handling
- Rate limiting (basic)
- Logging
- Auto-generated documentation (Swagger UI)

**Deliverables:**

- FastAPI application
- API documentation
- Example curl commands and Postman collection
- Basic API tests

---

### Phase 6: Web Interface (Week 5)

**Objectives:**

- Create user-friendly testing interface
- Visualize detection results
- Enable manual testing and validation

**Streamlit Interface Components:**

python

```python
import streamlit as st
import requests

st.title("🛡️ Phishing Email Detector")
st.markdown("Analyze emails for phishing indicators using AI")

# Input section
st.header("Email Details")
sender = st.text_input("Sender Email", "sender@example.com")
subject = st.text_input("Subject Line", "")
body = st.text_area("Email Body", height=200)
urls = st.text_area("URLs (one per line)", height=100)

# Detection button
if st.button("🔍 Analyze Email"):
    if body:
        # Parse URLs
        url_list = [u.strip() for u in urls.split('\n') if u.strip()]
        
        # Call API
        response = requests.post(
            "http://localhost:8000/api/v1/detect",
            json={
                "subject": subject,
                "body": body,
                "sender": sender,
                "urls": url_list
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Risk Score", f"{result['risk_score']:.2f}")
            with col2:
                st.metric("Email Score", f"{result['email_score']:.2f}")
            with col3:
                st.metric("Risk Level", result['risk_level'])
            
            # Risk visualization
            risk_color = "red" if result['risk_score'] > 0.7 else "orange" if result['risk_score'] > 0.4 else "green"
            st.progress(result['risk_score'])
            
            # Explanation
            st.subheader("Analysis Details")
            st.json(result['explanation'])
            
            # Indicators
            if result.get('indicators'):
                st.warning("⚠️ Detected Indicators:")
                for indicator in result['indicators']:
                    st.markdown(f"- {indicator}")
    else:
        st.error("Please enter email body")

# Sidebar with info
st.sidebar.header("About")
st.sidebar.info(
    "This tool uses advanced NLP and machine learning to detect "
    "phishing attempts. It analyzes email content, sender information, "
    "and embedded URLs."
)

st.sidebar.header("Example Phishing Indicators")
st.sidebar.markdown("""
- Urgent language and threats
- Requests for personal information
- Suspicious sender domains
- Mismatched or shortened URLs
- Poor grammar and spelling
- Generic greetings
""")
```

**Interface Features:**

- Clean, intuitive design
- Real-time analysis
- Visual risk scoring
- Detailed explanations
- Example phishing indicators
- Sample emails for testing

**Deliverables:**

- Streamlit application
- User documentation
- Demo video/screenshots

---

### Phase 7: Testing and Validation (Week 5-6)

**Objectives:**

- Comprehensive testing across scenarios
- Performance optimization
- Bug fixes and refinements

**Testing Strategy:**

1. **Unit Tests**
    - Feature extraction functions
    - Preprocessing utilities
    - Scoring algorithms
2. **Integration Tests**
    - End-to-end detection pipeline
    - API endpoint testing
    - Model inference validation
3. **Performance Tests**
    - Inference latency (<500ms per email)
    - Throughput testing
    - Memory usage monitoring
4. **Real-World Validation**
    - Test on fresh phishing examples
    - Verify against known legitimate emails
    - False positive analysis
    - Edge case testing
5. **Security Testing**
    - Input validation
    - Injection attack prevention
    - API security basics

**Test Cases:**

|Category|Test Scenarios|
|---|---|
|Legitimate|Corporate emails, newsletters, receipts, notifications|
|Phishing|Urgent requests, fake invoices, credential harvesting, CEO fraud|
|Edge Cases|Empty emails, very long texts, multiple URLs, special characters|
|Adversarial|Misspelled words, mixed languages, obfuscated URLs|

**Deliverables:**

- Test suite (pytest)
- Performance benchmarks
- Validation report
- Bug fix documentation

---

## 4. Dataset Guide

### 4.1 Recommended Datasets

**Primary Email Dataset:**

- **Kaggle Phishing Email Dataset**
    - Size: ~18,000 emails
    - Format: CSV with labels
    - Access: Free with Kaggle account

**Secondary Email Dataset:**

- **UTwente Phishing Emails**
    - Size: 2,000 labeled emails
    - Mix: Real and synthetic examples
    - Balanced classes

**URL Dataset:**

- **PhishTank**
    - Live database of phishing URLs
    - Updated daily
    - JSON/CSV export available
- **UCI Phishing Websites**
    - 11,000+ URLs with 30+ features
    - Pre-extracted features available

### 4.2 Data Preparation Steps

python

```python
# Example data loading and preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('phishing_emails.csv')

# Basic cleaning
df['text'] = df['subject'] + ' ' + df['body']
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.replace(r'<[^>]+>', '', regex=True)  # Remove HTML

# Extract URLs
df['urls'] = df['body'].str.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Train/val/test split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
```

---

## 5. Development Environment Setup

### 5.1 System Requirements

**Minimum:**

- Python 3.8+
- 8GB RAM
- 10GB disk space
- CPU-only capable

**Recommended:**

- Python 3.9+
- 16GB RAM
- GPU with 6GB+ VRAM (for faster training)
- 20GB disk space

### 5.2 Installation Guide

bash

````bash
# Create virtual environment
python -m venv phishing_detector_env
source phishing_detector_env/bin/activate  # On Windows: phishing_detector_env\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets
pip install scikit-learn pandas numpy
pip install fastapi uvicorn
pip install streamlit
pip install tldextract beautifulsoup4 requests
pip install pytest python-dotenv

# Save requirements
pip freeze > requirements.txt
```

### 5.3 Project Structure
```
phishing-detector/
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned and split data
│   └── external/             # PhishTank, WHOIS cache
├── models/
│   ├── email_classifier/     # Fine-tuned transformer
│   ├── url_classifier/       # URL model
│   └── checkpoints/          # Training checkpoints
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_email_model.ipynb
│   ├── 03_url_model.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── email_processor.py
│   │   └── url_extractor.py
│   ├── models/
│   │   ├── email_classifier.py
│   │   ├── url_classifier.py
│   │   └── detector.py
│   ├── features/
│   │   └── url_features.py
│   └── utils/
│       ├── config.py
│       └── helpers.py
├── api/
│   ├── main.py              # FastAPI app
│   ├── schemas.py           # Pydantic models
│   └── routers/
│       └── detection.py
├── web/
│   └── app.py               # Streamlit interface
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_api.py
├── config/
│   ├── config.yaml
│   └── model_config.json
├── requirements.txt
├── README.md
└── .gitignore
````

---

## 6. Model Training Guide

### 6.1 Email Classifier Training

**Step-by-Step Process:**

python

```python
# 1. Load and prepare data
from transformers import DistilBertTokenizer
from datasets import Dataset

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True,
        max_length=512
    )

train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)

# 2. Initialize model
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=2
)

# 3. Training configuration
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./models/email_classifier',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1'
)

# 4. Define metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
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

# 5. Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# 6. Save model
model.save_pretrained('./models/email_classifier/final')
tokenizer.save_pretrained('./models/email_classifier/final')
```

### 6.2 URL Classifier Training

python

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# Prepare URL features
X_train = url_train_df[feature_columns]
y_train = url_train_df['label']

# Train classifier
url_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    class_weight='balanced'
)

url_model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report
y_pred = url_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(url_model, './models/url_classifier/rf_model.pkl')
```

### 6.3 Training Tips

**For Email Model:**

- Start with 3 epochs, adjust if overfitting/underfitting
- Use gradient accumulation if memory limited
- Monitor validation loss - stop if diverging
- Save multiple checkpoints

**For URL Model:**

- Try XGBoost if Random Forest underperforms
- Tune hyperparameters with GridSearchCV
- Handle class imbalance with SMOTE if needed
- Feature importance analysis to reduce features

---

## 7. Deployment Guide

### 7.1 Running Locally

**Start API Server:**

bash

```bash
cd phishing-detector
uvicorn api.main:app --reload --port 8000
```

**Start Web Interface:**

bash

```bash
streamlit run web/app.py --server.port 8501
```

**Access:**

- API: [http://localhost:8000](http://localhost:8000)
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Web UI: [http://localhost:8501](http://localhost:8501)

### 7.2 Docker Deployment (Optional)

dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

yaml

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=/app/models
  
  web:
    build: .
    command: streamlit run web/app.py
    ports:
      - "8501:8501"
    depends_on:
      - api
```

### 7.3 Cloud Deployment Options

**For MVP Testing:**

- **Hugging Face Spaces:** Free tier for Streamlit apps
- **Railway:** Simple deployment, free tier available
- **Render:** Easy deployment with free tier

**For Production:**

- AWS (EC2, Lambda, SageMaker)
- Google Cloud (Cloud Run, Vertex AI)
- Azure (App Service, ML Studio)

---

## 8. Evaluation and Metrics

### 8.1 Performance Metrics

**Primary Metrics:**

- **Accuracy:** Overall correctness
- **Precision:** Of flagged emails, % actually phishing (minimize false alarms)
- **Recall:** Of phishing emails, % caught (maximize detection)
- **F1-Score:** Harmonic mean of precision and recall
- **AUC-ROC:** Model's ability to distinguish classes

**Target MVP Metrics:**

|Metric|Target|Minimum Acceptable|
|---|---|---|
|Accuracy|>92%|>88%|
|Precision|>90%|>85%|
|Recall|>88%|>82%|
|F1-Score|>90%|>85%|
|Inference Time|<500ms|<1000ms|

### 8.2 Evaluation Code

python

```python
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Predictions
y_true = test_df['label'].values
y_pred = model.predict(test_df['text'].values)
y_proba = model.predict_proba(test_df['text'].values)[:, 1]

# Classification report
print(classification_report(y_true, y_pred, 
                          target_names=['Legitimate', 'Phishing']))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_proba)
auc = roc_auc_score(y_true, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')

# Error analysis
false_positives = test_df[(y_true == 0) & (y_pred == 1)]
false_negatives = test_df[(y_true == 1) & (y_pred == 0)]

print(f"\nFalse Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")
```

---

## 9. Documentation and Deliverables

### 9.1 Code Documentation

**Required Documentation:**

- README.md with project overview
- API documentation (auto-generated by FastAPI)
- Code comments for complex logic
- Docstrings for all functions/classes

**README Template:**

markdown

```markdown
# Phishing Email and URL Detector

AI-powered system to detect phishing emails using NLP and machine learning.

## Features
- Email text classification using DistilBERT
- URL risk analysis
- REST API for integration
- Web interface for testing

## Installation
[Installation steps]

## Usage
[Usage examples]

## API Reference
[API endpoints]

## Model Performance
- Accuracy: 93%
- Precision: 91%
- Recall: 89%

## Project Structure
[Directory layout]

## Contributing
[Guidelines]

## License
MIT
```

### 9.2 Final Deliverables Checklist

**Code:**

- [ ]  Source code repository (GitHub)
- [ ]  Requirements.txt
- [ ]  Configuration files
- [ ]  Trained models (saved checkpoints)

**Documentation:**

- [ ]  README.md
- [ ]  API documentation
- [ ]  User guide for web interface
- [ ]  Model training guide
- [ ]  Deployment instructions

**Analysis:**

- [ ]  EDA notebook
- [ ]  Model training notebooks
- [ ]  Evaluation report (PDF)
- [ ]  Performance visualizations

**Demo:**

- [ ]  Live demo (deployed or video)
- [ ]  Example inputs and outputs
- [ ]  Presentation slides (optional)

**Tests:**

- [ ]  Unit tests
- [ ]  Integration tests
- [ ]  Test coverage report