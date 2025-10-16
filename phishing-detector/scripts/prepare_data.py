"""
Data preparation script
"""

import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.email_processor import EmailProcessor


def prepare_email_data(
    input_path: str,
    output_dir: str,
    test_size: float = 0.3,
    random_state: int = 42
):
    """
    Prepare email dataset for training
    
    Args:
        input_path: Path to raw email CSV
        output_dir: Output directory for processed data
        test_size: Proportion of data for test/validation
        random_state: Random seed
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    print(f"Loaded {len(df)} emails")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Initialize processor
    processor = EmailProcessor()
    
    # Preprocess text
    print("Preprocessing email text...")
    df['processed_text'] = df.apply(
        lambda row: processor.preprocess(
            f"{row['subject']} {row['body']}"
        ),
        axis=1
    )
    
    # Extract features
    print("Extracting features...")
    features_df = df.apply(
        lambda row: pd.Series(
            processor.extract_features(
                row['subject'],
                row['body'],
                row['sender']
            )
        ),
        axis=1
    )
    
    # Combine with original data
    df = pd.concat([df, features_df], axis=1)
    
    # Split data
    print("Splitting data...")
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Saved processed data to {output_dir}")


if __name__ == "__main__":
    # Example usage
    input_file = "data/raw/sample_emails.csv"
    output_dir = "data/processed"
    
    if os.path.exists(input_file):
        prepare_email_data(input_file, output_dir)
    else:
        print(f"Input file not found: {input_file}")
        print("Please add your email dataset to data/raw/")
