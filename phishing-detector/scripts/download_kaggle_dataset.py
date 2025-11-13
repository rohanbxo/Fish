"""
Download and prepare the Phishing Email Dataset from Kaggle
Dataset: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

Supports multiple download methods:
1. KaggleHub (newer, simpler)
2. Kaggle API (traditional, more control)
3. Manual download
"""

import os
import pandas as pd
from pathlib import Path
import re
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

class PhishingDatasetDownloader:
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_name = "naserabdullahalam/phishing-email-dataset"
    
    def download_with_kagglehub(self):
        """
        METHOD 1: Download using KaggleHub (New Way)
        Simplest method - just works!
        """
        try:
            import kagglehub
            
            logger.info("=" * 70)
            logger.info("📥 DOWNLOADING DATASET USING KAGGLEHUB")
            logger.info("=" * 70)
            
            logger.info(f"Dataset: {self.dataset_name}")
            logger.info("Please wait, this may take a few minutes...")
            
            # Download to Kaggle's cache
            cache_path = kagglehub.dataset_download(self.dataset_name)
            
            logger.info(f"✅ Downloaded to: {cache_path}")
            
            # Copy files to our project directory
            import shutil
            cache_path = Path(cache_path)
            
            copied_files = []
            for file in cache_path.glob('*.csv'):
                dest = self.raw_dir / file.name
                logger.info(f"📋 Copying {file.name} to {dest}")
                shutil.copy2(file, dest)
                copied_files.append(file.name)
            
            if copied_files:
                logger.info(f"\n✅ Successfully copied {len(copied_files)} file(s) to {self.raw_dir}")
                return True
            else:
                logger.warning("⚠️  No CSV files found in download")
                return False
            
        except ImportError:
            logger.warning("⚠️  kagglehub not installed")
            logger.info("Install with: pip install kagglehub")
            return False
        except Exception as e:
            logger.warning(f"⚠️  KaggleHub failed: {e}")
            return False
    
    def download_with_kaggle_api(self):
        """
        METHOD 2: Download using Kaggle API (Traditional Way)
        Requires kaggle.json setup
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            logger.info("=" * 70)
            logger.info("📥 DOWNLOADING DATASET USING KAGGLE API")
            logger.info("=" * 70)
            
            logger.info("Authenticating with Kaggle API...")
            api = KaggleApi()
            api.authenticate()
            
            logger.info(f"Dataset: {self.dataset_name}")
            logger.info("Downloading and extracting...")
            
            # Download directly to our directory
            api.dataset_download_files(
                self.dataset_name,
                path=str(self.raw_dir),
                unzip=True
            )
            
            # List downloaded files
            files = list(self.raw_dir.glob('*.csv'))
            
            logger.info(f"\n✅ Dataset downloaded to: {self.raw_dir}")
            logger.info(f"📁 Downloaded files:")
            for f in files:
                size_mb = f.stat().st_size / 1024 / 1024
                logger.info(f"   • {f.name} ({size_mb:.2f} MB)")
            
            return True
            
        except ImportError:
            logger.warning("⚠️  kaggle package not installed")
            logger.info("Install with: pip install kaggle")
            return False
        except OSError as e:
            if "Could not find kaggle.json" in str(e):
                logger.warning("⚠️  kaggle.json not found!")
                logger.info("\n📝 Setup Instructions:")
                logger.info("   1. Go to https://www.kaggle.com/settings")
                logger.info("   2. Click 'Create New API Token'")
                logger.info("   3. Save kaggle.json to:")
                logger.info(f"      C:\\Users\\{os.getenv('USERNAME')}\\.kaggle\\kaggle.json")
                return False
            else:
                logger.warning(f"⚠️  Error: {e}")
                return False
        except Exception as e:
            logger.warning(f"⚠️  Kaggle API failed: {e}")
            return False
    
    def show_manual_instructions(self):
        """Show manual download instructions"""
        logger.info("=" * 70)
        logger.info("📝 MANUAL DOWNLOAD INSTRUCTIONS")
        logger.info("=" * 70)
        logger.info("\nIf automatic download fails, download manually:")
        logger.info(f"\n1. Visit: https://www.kaggle.com/datasets/{self.dataset_name}")
        logger.info("2. Click the 'Download' button (requires Kaggle account)")
        logger.info("3. Extract the ZIP file")
        logger.info(f"4. Copy the CSV file(s) to: {self.raw_dir.absolute()}")
        logger.info("5. Run this script again with: --skip-download\n")
    
    def auto_download(self):
        """Automatically try both download methods"""
        logger.info("🚀 Starting automatic download...\n")
        
        # Check if files already exist
        existing_files = list(self.raw_dir.glob('*.csv'))
        if existing_files:
            logger.info(f"📁 Found existing CSV file(s) in {self.raw_dir}:")
            for f in existing_files:
                logger.info(f"   • {f.name}")
            
            while True:
                response = input("\nUse existing files? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    logger.info("✅ Using existing files")
                    return True
                elif response in ['n', 'no']:
                    logger.info("Proceeding with download...")
                    break
                else:
                    print("Please enter 'y' or 'n'")
        
        # Try Method 1: KaggleHub (simpler, doesn't need setup)
        logger.info("\n🔄 Trying Method 1: KaggleHub...")
        if self.download_with_kagglehub():
            return True
        
        logger.info("\n" + "=" * 70)
        logger.info("🔄 Method 1 failed. Trying Method 2: Kaggle API...")
        logger.info("=" * 70 + "\n")
        
        # Try Method 2: Kaggle API
        if self.download_with_kaggle_api():
            return True
        
        # Both methods failed
        logger.warning("\n❌ Automatic download failed")
        self.show_manual_instructions()
        return False
    
    def load_and_explore(self):
        """Load and explore the downloaded dataset"""
        csv_files = list(self.raw_dir.glob('*.csv'))
        
        if not csv_files:
            logger.error("❌ No CSV files found in data/raw/")
            logger.info("Please download the dataset first")
            return None
        
        # Use the first CSV file (or the largest one)
        csv_path = max(csv_files, key=lambda f: f.stat().st_size)
        
        logger.info("=" * 70)
        logger.info("📖 LOADING DATASET")
        logger.info("=" * 70)
        logger.info(f"File: {csv_path.name}")
        logger.info(f"Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        try:
            df = pd.read_csv(csv_path)
            
            logger.info("\n📊 DATASET OVERVIEW")
            logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            logger.info(f"\nColumns: {df.columns.tolist()}")
            
            logger.info("\nFirst 3 rows:")
            print(df.head(3).to_string())
            
            logger.info(f"\nData types:")
            print(df.dtypes.to_string())
            
            logger.info(f"\nMissing values:")
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(missing[missing > 0].to_string())
            else:
                logger.info("No missing values ✅")
            
            # Find and show class distribution
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['label', 'type', 'class', 'category']):
                    logger.info(f"\n📈 Class Distribution ({col}):")
                    print(df[col].value_counts().to_string())
                    break
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV: {e}")
            return None
    
    def prepare_data(self, df):
        """Clean and prepare the dataset"""
        if df is None:
            return None
        
        logger.info("\n" + "=" * 70)
        logger.info("🔧 PREPARING DATA")
        logger.info("=" * 70)
        
        # Detect text and label columns
        text_col = None
        label_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            # Find text column
            if any(keyword in col_lower for keyword in ['email', 'text', 'body', 'message', 'content']):
                if text_col is None:  # Take first match
                    text_col = col
            # Find label column
            if any(keyword in col_lower for keyword in ['label', 'type', 'class', 'category']):
                if label_col is None:  # Take first match
                    label_col = col
        
        if not text_col:
            logger.error("❌ Could not find text column")
            logger.info(f"Available columns: {df.columns.tolist()}")
            logger.info("Please check the dataset structure")
            return None
        
        if not label_col:
            logger.error("❌ Could not find label column")
            logger.info(f"Available columns: {df.columns.tolist()}")
            logger.info("Please check the dataset structure")
            return None
        
        logger.info(f"✅ Text column: '{text_col}'")
        logger.info(f"✅ Label column: '{label_col}'")
        
        # Create clean dataframe
        df_clean = df[[text_col, label_col]].copy()
        df_clean.columns = ['text', 'label']
        
        # Clean text
        logger.info("\n🧹 Cleaning text data...")
        df_clean['text'] = df_clean['text'].astype(str)
        df_clean['text'] = df_clean['text'].str.strip()
        
        # Standardize labels to binary (0=legitimate, 1=phishing)
        logger.info("🏷️  Standardizing labels...")
        unique_labels = df_clean['label'].unique()
        logger.info(f"Original labels: {unique_labels}")
        
        label_map = {}
        for label in unique_labels:
            label_str = str(label).lower()
            # Legitimate/Safe emails
            if any(word in label_str for word in ['safe', 'legitimate', 'ham', 'normal', 'benign', '0']):
                label_map[label] = 0
            # Phishing/Malicious emails
            elif any(word in label_str for word in ['phishing', 'spam', 'malicious', 'suspicious', '1']):
                label_map[label] = 1
        
        df_clean['label'] = df_clean['label'].map(label_map)
        
        # Remove unmapped labels
        unmapped = df_clean['label'].isna().sum()
        if unmapped > 0:
            logger.warning(f"⚠️  Removing {unmapped} rows with unmapped labels")
            df_clean = df_clean.dropna(subset=['label'])
        
        df_clean['label'] = df_clean['label'].astype(int)
        logger.info(f"✅ Labels: 0 (Legitimate), 1 (Phishing)")
        
        # Remove duplicates
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=['text'])
        removed = before - len(df_clean)
        if removed > 0:
            logger.info(f"🗑️  Removed {removed:,} duplicate emails")
        
        # Remove very short emails (likely errors)
        before = len(df_clean)
        df_clean = df_clean[df_clean['text'].str.len() >= 10]
        removed = before - len(df_clean)
        if removed > 0:
            logger.info(f"🗑️  Removed {removed:,} too-short emails")
        
        # Add metadata columns
        logger.info("📊 Adding metadata...")
        df_clean['text_length'] = df_clean['text'].str.len()
        df_clean['word_count'] = df_clean['text'].str.split().str.len()
        
        # Extract URLs
        logger.info("🔗 Extracting URLs...")
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        df_clean['urls'] = df_clean['text'].str.findall(url_pattern)
        df_clean['url_count'] = df_clean['urls'].str.len()
        
        # Show final statistics
        logger.info("\n" + "=" * 70)
        logger.info("✅ CLEANED DATASET STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total emails: {len(df_clean):,}")
        logger.info(f"\n📊 Class Distribution:")
        logger.info(f"   Legitimate (0): {(df_clean['label']==0).sum():,} ({(df_clean['label']==0).sum()/len(df_clean)*100:.1f}%)")
        logger.info(f"   Phishing (1):   {(df_clean['label']==1).sum():,} ({(df_clean['label']==1).sum()/len(df_clean)*100:.1f}%)")
        
        logger.info(f"\n📏 Text Statistics:")
        logger.info(f"   Avg length: {df_clean['text_length'].mean():.0f} characters")
        logger.info(f"   Avg words:  {df_clean['word_count'].mean():.0f} words")
        logger.info(f"   Min length: {df_clean['text_length'].min():.0f} characters")
        logger.info(f"   Max length: {df_clean['text_length'].max():.0f} characters")
        
        emails_with_urls = (df_clean['url_count'] > 0).sum()
        logger.info(f"\n🔗 URL Statistics:")
        logger.info(f"   Emails with URLs: {emails_with_urls:,} ({emails_with_urls/len(df_clean)*100:.1f}%)")
        logger.info(f"   Avg URLs per email: {df_clean['url_count'].mean():.2f}")
        
        return df_clean
    
    def split_and_save(self, df):
        """Split into train/val/test sets and save"""
        from sklearn.model_selection import train_test_split
        
        logger.info("\n" + "=" * 70)
        logger.info("✂️  SPLITTING DATASET")
        logger.info("=" * 70)
        
        # Split: 70% train, 15% validation, 15% test
        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            stratify=df['label'],
            random_state=42
        )
        
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df['label'],
            random_state=42
        )
        
        logger.info(f"Train set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"   - Legitimate: {(train_df['label']==0).sum():,}")
        logger.info(f"   - Phishing:   {(train_df['label']==1).sum():,}")
        
        logger.info(f"\nValidation set: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"   - Legitimate: {(val_df['label']==0).sum():,}")
        logger.info(f"   - Phishing:   {(val_df['label']==1).sum():,}")
        
        logger.info(f"\nTest set: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        logger.info(f"   - Legitimate: {(test_df['label']==0).sum():,}")
        logger.info(f"   - Phishing:   {(test_df['label']==1).sum():,}")
        
        # Save to CSV
        logger.info(f"\n💾 Saving to {self.processed_dir}/")
        
        train_df.to_csv(self.processed_dir / 'train.csv', index=False)
        logger.info("   ✅ train.csv")
        
        val_df.to_csv(self.processed_dir / 'val.csv', index=False)
        logger.info("   ✅ val.csv")
        
        test_df.to_csv(self.processed_dir / 'test.csv', index=False)
        logger.info("   ✅ test.csv")
        
        df.to_csv(self.processed_dir / 'full_cleaned.csv', index=False)
        logger.info("   ✅ full_cleaned.csv")
        
        return train_df, val_df, test_df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download and prepare Kaggle phishing email dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/download_kaggle_dataset.py                    # Auto download
  python scripts/download_kaggle_dataset.py --method kagglehub  # Use KaggleHub
  python scripts/download_kaggle_dataset.py --method api        # Use Kaggle API
  python scripts/download_kaggle_dataset.py --skip-download     # Prepare existing data
        """
    )
    
    parser.add_argument(
        '--method',
        choices=['kagglehub', 'api', 'auto'],
        default='auto',
        help='Download method (default: auto - tries both)'
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download, only prepare existing data'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🛡️  PHISHING EMAIL DATASET PREPARATION")
    print("=" * 70)
    print()
    
    downloader = PhishingDatasetDownloader()
    
    # Download phase
    if not args.skip_download:
        if args.method == 'kagglehub':
            success = downloader.download_with_kagglehub()
        elif args.method == 'api':
            success = downloader.download_with_kaggle_api()
        else:  # auto
            success = downloader.auto_download()
        
        if not success:
            logger.error("\n❌ Download failed")
            logger.info("Please download manually or try a different method")
            sys.exit(1)
    else:
        logger.info("⏭️  Skipping download (using existing files)")
    
    # Load phase
    df = downloader.load_and_explore()
    if df is None:
        sys.exit(1)
    
    # Prepare phase
    df = downloader.prepare_data(df)
    if df is None:
        sys.exit(1)
    
    # Split and save
    train_df, val_df, test_df = downloader.split_and_save(df)
    
    # Success message
    print("\n" + "=" * 70)
    print("🎉 SUCCESS! Dataset is ready for training")
    print("=" * 70)
    print(f"\n📁 Processed data saved to: data/processed/")
    print(f"\n🚀 Next steps:")
    print(f"   1. Review the data: data/processed/train.csv")
    print(f"   2. Train model: python scripts/train_model.py")
    print(f"   3. Start API: python api/main.py")
    print(f"   4. Start Web UI: streamlit run web/app.py")
    print()


if __name__ == "__main__":
    main()
