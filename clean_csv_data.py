#!/usr/bin/env python3
"""
CSV Data Cleaner Tool
Cleans and standardizes analysis_log.csv for better dashboard readability
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

def clean_csv_data(input_csv, output_csv=None):
    """Clean and standardize CSV data for better readability"""
    
    if output_csv is None:
        output_csv = input_csv.replace('.csv', '_clean.csv')
    
    print(f"🧹 Cleaning CSV data: {input_csv} -> {output_csv}")
    
    # Load the CSV
    try:
        df = pd.read_csv(input_csv)
        print(f"✅ Loaded: {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    # Show original data sample
    print(f"\n📊 Original data sample:")
    print(df.head(3).to_string())
    
    # 1. Clean timestamps
    print(f"\n🕒 Cleaning timestamps...")
    df['ts_utc'] = df['ts_utc'].str.strip()
    df['ts_utc'] = pd.to_datetime(df['ts_utc'], format='ISO8601')
    
    # 2. Clean speaker names
    print(f"👥 Cleaning speaker names...")
    df['speaker'] = df['speaker'].str.strip()
    
    # Extract clean speaker names (remove bracketed match info)
    def clean_speaker_name(speaker):
        if '[' in speaker:
            return speaker.split('[')[0].strip()
        return speaker.strip()
    
    df['speaker_clean'] = df['speaker'].apply(clean_speaker_name)
    
    # 3. Clean emotion predictions
    print(f"😊 Cleaning emotion predictions...")
    df['pred_class'] = df['pred_class'].str.strip().str.lower()
    
    # Standardize emotion names
    emotion_mapping = {
        'happy': 'happy',
        'angry': 'angry', 
        'fear': 'fear',
        'calm': 'calm',
        'sad': 'sad',
        'surprise': 'surprise'
    }
    df['pred_class'] = df['pred_class'].map(emotion_mapping).fillna(df['pred_class'])
    
    # 4. Clean text transcripts
    print(f"📝 Cleaning text transcripts...")
    df['text'] = df['text'].astype(str).str.strip()
    
    # Remove excessive spaces and clean Thai text
    def clean_text(text):
        if pd.isna(text) or text == 'nan':
            return ''
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing spaces
        text = text.strip()
        return text
    
    df['text_clean'] = df['text'].apply(clean_text)
    
    # 5. Clean confidence scores
    print(f"🎯 Cleaning confidence scores...")
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df['confidence'] = df['confidence'].fillna(0.0)
    df['confidence'] = df['confidence'].clip(0.0, 1.0)
    
    # 6. Clean entropy
    print(f"📊 Cleaning entropy values...")
    df['entropy'] = pd.to_numeric(df['entropy'], errors='coerce')
    df['entropy'] = df['entropy'].fillna(0.0)
    
    # 7. Clean RMS values
    print(f"🔊 Cleaning RMS values...")
    df['rms'] = pd.to_numeric(df['rms'], errors='coerce')
    df['rms'] = df['rms'].fillna(0.0)
    
    # 8. Clean probability columns
    print(f"📈 Cleaning probability columns...")
    prob_cols = [col for col in df.columns if col.startswith('prob_')]
    for col in prob_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(0.0)
        df[col] = df[col].clip(0.0, 1.0)
    
    # 9. Add data quality metrics
    print(f"🔍 Adding data quality metrics...")
    
    # Text quality
    df['text_length'] = df['text_clean'].str.len()
    df['has_text'] = (df['text_length'] > 0).astype(int)
    
    # Confidence quality
    df['high_confidence'] = (df['confidence'] >= 0.7).astype(int)
    df['low_confidence'] = (df['confidence'] < 0.5).astype(int)
    
    # Speaker consistency
    df['speaker_changed'] = (df['speaker_clean'] != df['speaker_clean'].shift()).astype(int)
    
    # 10. Sort by timestamp
    print(f"🔄 Sorting by timestamp...")
    df = df.sort_values('ts_utc').reset_index(drop=True)
    
    # 11. Select and reorder columns for clarity
    print(f"📋 Reorganizing columns...")
    
    # Define clean column order
    clean_columns = [
        'ts_utc',
        'file', 
        'speaker_clean',
        'pred_class',
        'confidence',
        'entropy',
        'rms',
        'text_clean',
        'prob_happy',
        'prob_angry', 
        'prob_fear',
        'prob_calm',
        'prob_sad',
        'prob_surprise',
        'text_length',
        'has_text',
        'high_confidence',
        'low_confidence'
    ]
    
    # Only include columns that exist
    final_columns = [col for col in clean_columns if col in df.columns]
    
    # Create final clean dataframe
    clean_df = df[final_columns].copy()
    
    # Rename columns for clarity
    clean_df = clean_df.rename(columns={
        'speaker_clean': 'speaker',
        'text_clean': 'text'
    })
    
    # 12. Add summary statistics
    print(f"\n📊 Data Quality Summary:")
    print(f"   • Total rows: {len(clean_df)}")
    print(f"   • Unique speakers: {clean_df['speaker'].nunique()}")
    print(f"   • Unique emotions: {clean_df['pred_class'].nunique()}")
    print(f"   • Average confidence: {clean_df['confidence'].mean():.3f}")
    print(f"   • High confidence segments: {clean_df['high_confidence'].sum()} ({clean_df['high_confidence'].mean():.1%})")
    print(f"   • Segments with text: {clean_df['has_text'].sum()} ({clean_df['has_text'].mean():.1%})")
    print(f"   • Date range: {clean_df['ts_utc'].min()} to {clean_df['ts_utc'].max()}")
    
    # Show emotion distribution
    print(f"\n😊 Emotion Distribution:")
    emotion_counts = clean_df['pred_class'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   • {emotion}: {count} ({count/len(clean_df):.1%})")
    
    # Show speaker distribution (top 10)
    print(f"\n👥 Top 10 Speakers:")
    speaker_counts = clean_df['speaker'].value_counts().head(10)
    for speaker, count in speaker_counts.items():
        print(f"   • {speaker}: {count} segments")
    
    # 13. Save cleaned data
    print(f"\n💾 Saving cleaned data...")
    clean_df.to_csv(output_csv, index=False)
    
    # Show cleaned data sample
    print(f"\n📊 Cleaned data sample:")
    print(clean_df.head(3).to_string())
    
    print(f"\n✅ Cleaning complete!")
    print(f"   • Original file: {input_csv}")
    print(f"   • Cleaned file: {output_csv}")
    print(f"   • File size: {Path(output_csv).stat().st_size / 1024:.1f} KB")
    
    return True

def create_backup(original_csv):
    """Create a backup of the original CSV"""
    backup_file = original_csv.replace('.csv', '_backup.csv')
    
    if Path(original_csv).exists():
        import shutil
        shutil.copy2(original_csv, backup_file)
        print(f"📋 Backup created: {backup_file}")
        return backup_file
    else:
        print(f"❌ Original file not found: {original_csv}")
        return None

def main():
    import sys
    
    # Get CSV file from command line or use default
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "analysis_log.csv"
    
    print("🧹 CSV Data Cleaner Tool")
    print("=" * 40)
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"❌ File not found: {csv_file}")
        return
    
    # Create backup
    backup_file = create_backup(csv_file)
    
    # Clean the data
    success = clean_csv_data(csv_file)
    
    if success:
        print(f"\n🎉 CSV data cleaned successfully!")
        print(f"\n📝 Next steps:")
        print(f"   1. Update dashboard to use: {csv_file.replace('.csv', '_clean.csv')}")
        print(f"   2. Or replace original: cp {csv_file.replace('.csv', '_clean.csv')} {csv_file}")
        print(f"   3. Restart the dashboard")
        
        if backup_file:
            print(f"   4. Original file backed up to: {backup_file}")
    else:
        print(f"\n❌ Cleaning failed!")

if __name__ == "__main__":
    main()
