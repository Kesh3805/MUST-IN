"""
Verify dataset format and quality for MUST-IN training.

Usage:
    python scripts/validate_dataset.py --data data/raw/sample_dataset.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def validate_dataset(file_path: str) -> bool:
    """
    Validate dataset format and quality.
    
    Args:
        file_path: Path to CSV dataset
    
    Returns:
        bool: True if valid, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Validating Dataset: {file_path}")
    print(f"{'='*60}\n")
    
    try:
        # Load dataset
        print("📂 Loading dataset...")
        df = pd.read_csv(file_path)
        print(f"✓ Dataset loaded: {len(df):,} samples\n")
        
        # Check required columns
        print("📋 Checking columns...")
        required_cols = ['text', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            print(f"  Found columns: {list(df.columns)}")
            return False
        
        print(f"✓ Required columns present: {', '.join(required_cols)}")
        
        # Check optional columns
        optional_cols = ['language', 'platform', 'source']
        present_optional = [col for col in optional_cols if col in df.columns]
        if present_optional:
            print(f"✓ Optional columns present: {', '.join(present_optional)}")
        print()
        
        # Check for empty values
        print("🔍 Checking data quality...")
        
        empty_text = df['text'].isna().sum()
        if empty_text > 0:
            print(f"⚠ Found {empty_text} empty text entries ({empty_text/len(df)*100:.1f}%)")
        else:
            print(f"✓ No empty text entries")
        
        empty_labels = df['label'].isna().sum()
        if empty_labels > 0:
            print(f"✗ Found {empty_labels} empty labels ({empty_labels/len(df)*100:.1f}%)")
            return False
        else:
            print(f"✓ No empty labels")
        print()
        
        # Check label distribution
        print("📊 Label distribution:")
        label_counts = df['label'].value_counts().sort_index()
        
        label_names = {0: 'Neutral', 1: 'Offensive', 2: 'Hate'}
        total = len(df)
        
        for label_id, count in label_counts.items():
            percentage = count / total * 100
            label_name = label_names.get(label_id, f'Unknown({label_id})')
            bar = '█' * int(percentage / 2)
            print(f"  {label_name:<12} {count:>6} ({percentage:>5.1f}%) {bar}")
        
        # Check for invalid labels
        valid_labels = {0, 1, 2}
        invalid_labels = set(df['label'].unique()) - valid_labels
        if invalid_labels:
            print(f"\n✗ Invalid labels found: {invalid_labels}")
            print(f"  Valid labels are: {valid_labels}")
            return False
        
        # Check label balance
        min_count = label_counts.min()
        max_count = label_counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print()
        if imbalance_ratio > 10:
            print(f"⚠ Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
            print(f"  Consider rebalancing or using class weights")
        elif imbalance_ratio > 3:
            print(f"⚠ Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)")
        else:
            print(f"✓ Balanced label distribution (ratio: {imbalance_ratio:.1f}:1)")
        print()
        
        # Check text lengths
        print("📏 Text length statistics:")
        df['text_length'] = df['text'].astype(str).str.len()
        
        print(f"  Min length: {df['text_length'].min()} chars")
        print(f"  Max length: {df['text_length'].max()} chars")
        print(f"  Mean length: {df['text_length'].mean():.1f} chars")
        print(f"  Median length: {df['text_length'].median():.1f} chars")
        
        # Check for very short/long texts
        very_short = (df['text_length'] < 5).sum()
        very_long = (df['text_length'] > 500).sum()
        
        if very_short > 0:
            print(f"  ⚠ {very_short} very short texts (<5 chars)")
        if very_long > 0:
            print(f"  ⚠ {very_long} very long texts (>500 chars)")
        print()
        
        # Check language distribution (if available)
        if 'language' in df.columns:
            print("🌐 Language distribution:")
            lang_counts = df['language'].value_counts()
            
            for lang, count in lang_counts.head(10).items():
                percentage = count / total * 100
                print(f"  {lang:<15} {count:>6} ({percentage:>5.1f}%)")
            
            if len(lang_counts) > 10:
                print(f"  ... and {len(lang_counts) - 10} more languages")
            print()
        
        # Check for duplicates
        print("🔄 Checking duplicates...")
        duplicates = df.duplicated(subset=['text']).sum()
        if duplicates > 0:
            print(f"⚠ Found {duplicates} duplicate texts ({duplicates/len(df)*100:.1f}%)")
            print(f"  Consider removing duplicates before training")
        else:
            print(f"✓ No duplicate texts found")
        print()
        
        # Sample texts
        print("📝 Sample texts:")
        for label_id in [0, 1, 2]:
            samples = df[df['label'] == label_id]['text'].head(1)
            if not samples.empty:
                label_name = label_names[label_id]
                text = samples.iloc[0][:80]
                print(f"  {label_name:<12}: {text}...")
        print()
        
        # Final verdict
        print(f"{'='*60}")
        print("✅ Dataset validation passed!")
        print(f"{'='*60}\n")
        
        print("📈 Dataset Summary:")
        print(f"  Total samples: {len(df):,}")
        print(f"  Neutral: {label_counts.get(0, 0):,}")
        print(f"  Offensive: {label_counts.get(1, 0):,}")
        print(f"  Hate: {label_counts.get(2, 0):,}")
        print(f"  Avg text length: {df['text_length'].mean():.0f} chars")
        if 'language' in df.columns:
            print(f"  Languages: {len(df['language'].unique())}")
        print()
        
        print("✅ Dataset ready for training!\n")
        print("Next steps:")
        print("  1. Train model: python main.py --run-dl --save-models")
        print("  2. See TRAINING_WORKFLOW.md for complete guide")
        
        return True
        
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return False
    except pd.errors.EmptyDataError:
        print(f"✗ Empty CSV file")
        return False
    except Exception as e:
        print(f"✗ Error validating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate MUST-IN training dataset"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/raw/sample_dataset.csv",
        help="Path to dataset CSV file"
    )
    
    args = parser.parse_args()
    
    # Validate
    is_valid = validate_dataset(args.data)
    
    # Exit code
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
