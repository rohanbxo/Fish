# 🚀 Quick Start Guide - Phishing Detector with Kaggle Dataset

## Step 1: Install Dependencies

```powershell
cd "C:\Users\Rohan Burugapalli\Desktop\Fish\phishing-detector"

# Activate virtual environment (if not already activated)
.\venv\Scripts\Activate

# Install required packages
pip install kagglehub kaggle pandas scikit-learn
```

## Step 2: Download the Dataset

### Option A: Automatic (Recommended)
```powershell
python scripts\download_kaggle_dataset.py
```

This will automatically try to download using the easiest method available.

### Option B: Choose Your Method

**Using KaggleHub (Simplest)**
```powershell
# Install kagglehub
pip install kagglehub

# Download
python scripts\download_kaggle_dataset.py --method kagglehub
```

**Using Kaggle API (More Control)**
```powershell
# Install kaggle
pip install kaggle

# Setup credentials (one-time)
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token"
# 3. Save kaggle.json to: C:\Users\Rohan Burugapalli\.kaggle\kaggle.json

# Download
python scripts\download_kaggle_dataset.py --method api
```

**Manual Download**
```powershell
# 1. Visit: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
# 2. Click Download
# 3. Extract CSV to: phishing-detector\data\raw\
# 4. Run:
python scripts\download_kaggle_dataset.py --skip-download
```

## Step 3: Verify Data is Ready

After successful download, you should see:
```
data/
  processed/
    ├── train.csv       (~13,000 emails)
    ├── val.csv         (~1,900 emails)
    ├── test.csv        (~3,700 emails)
    └── full_cleaned.csv
```

## Step 4: Train the Model (Optional - can use pretrained)

```powershell
python scripts\train_model.py
```

## Step 5: Start the Application

**Terminal 1: Start API**
```powershell
cd api
python main.py
```

**Terminal 2: Start Web Interface**
```powershell
streamlit run web\app.py
```

## Step 6: Test It!

1. **Web Interface**: Open http://localhost:8501
2. **API**: Open http://localhost:8000/docs

## Troubleshooting

### If download fails:
```powershell
# Install both packages
pip install kagglehub kaggle

# Try automatic method
python scripts\download_kaggle_dataset.py
```

### If you get "kaggle.json not found":
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save the downloaded `kaggle.json` to: `C:\Users\Rohan Burugapalli\.kaggle\kaggle.json`

### If you want to use existing CSV:
```powershell
# Put your CSV in: data/raw/
# Then run:
python scripts\download_kaggle_dataset.py --skip-download
```

## What the Download Script Does

✅ Downloads ~18,600 phishing emails from Kaggle  
✅ Cleans and standardizes the data  
✅ Extracts URLs from emails  
✅ Splits into train/val/test (70%/15%/15%)  
✅ Saves processed data ready for training  
✅ Generates statistics and metadata  

## Expected Output

```
📊 DATASET OVERVIEW
Shape: 18,650 rows × 2 columns
Columns: ['text', 'label']

📈 Class Distribution:
   Legitimate (0): 9,325 (50.0%)
   Phishing (1):   9,325 (50.0%)

📏 Text Statistics:
   Avg length: 1,245 characters
   Avg words:  234 words

🔗 URL Statistics:
   Emails with URLs: 12,455 (66.8%)
   Avg URLs per email: 2.3

✂️  SPLITTING DATASET
Train set: 13,055 samples (70.0%)
Val set:   2,798 samples (15.0%)
Test set:  2,797 samples (15.0%)

💾 Saved to data/processed/
   ✅ train.csv
   ✅ val.csv
   ✅ test.csv
   ✅ full_cleaned.csv

🎉 SUCCESS! Dataset is ready for training
```

## Next Steps After Data is Ready

1. **Explore the data**:
   ```powershell
   # Open in Excel or Python
   import pandas as pd
   df = pd.read_csv('data/processed/train.csv')
   print(df.head())
   ```

2. **Train the model**:
   ```powershell
   python scripts/train_model.py
   ```

3. **Evaluate performance**:
   ```powershell
   python scripts/evaluate_model.py
   ```

4. **Launch the application**:
   ```powershell
   # Terminal 1
   python api/main.py
   
   # Terminal 2
   streamlit run web/app.py
   ```

## Quick Command Reference

```powershell
# Download dataset
python scripts\download_kaggle_dataset.py

# Train model
python scripts\train_model.py

# Evaluate model
python scripts\evaluate_model.py

# Run tests
pytest tests\ -v

# Start API
python api\main.py

# Start Web UI
streamlit run web\app.py

# Run all tests with coverage
pytest tests\ --cov=src --cov-report=html
```

## Need Help?

- Check logs in `logs/` directory
- Review `data/processed/` to verify data
- Run with `--help` flag: `python scripts/download_kaggle_dataset.py --help`
- Check API docs: http://localhost:8000/docs
