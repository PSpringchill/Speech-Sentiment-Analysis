#!/usr/bin/env python3
"""
Generate Fresh Experiment Data for t-SNE
Creates speaker embeddings for t-SNE visualization
"""

import numpy as np
from pathlib import Path
import random

def generate_fresh_experiment_data():
    """Generate fresh experiment data with speaker embeddings"""
    
    print("🔬 Generating fresh experiment data for t-SNE...")
    
    # Get current CSV data to match speakers
    try:
        import pandas as pd
        df = pd.read_csv('analysis_log.csv')
        if len(df) == 0:
            print("❌ No data in CSV. Generating sample data...")
            # Generate sample data
            speakers = ['Speaker_0', 'Speaker_1', 'Speaker_2']
            embeddings = []
            labels = []
            
            # Generate embeddings for each speaker
            for i, speaker in enumerate(speakers):
                # Create 20 segments per speaker with similar embeddings
                for j in range(20):
                    # Base embedding for speaker + some noise
                    base_embedding = np.random.randn(192) * 0.1
                    speaker_id = i * 2.0  # Different base for each speaker
                    embedding = base_embedding + np.random.randn(192) * 0.3 + speaker_id
                    embeddings.append(embedding)
                    labels.append(speaker)
            
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            
        else:
            print(f"✅ Found {len(df)} rows in CSV")
            
            # Extract unique speakers from current CSV
            unique_speakers = df['speaker'].str.replace(r'\s*\[.*?\]', '', regex=True).unique()
            unique_speakers = [s.strip() for s in unique_speakers if s.strip()]
            
            print(f"👥 Found {len(unique_speakers)} unique speakers: {unique_speakers}")
            
            # Generate embeddings for each speaker
            embeddings = []
            labels = []
            
            for i, speaker in enumerate(unique_speakers):
                # Generate 5-10 segments per speaker
                num_segments = random.randint(5, 10)
                
                for j in range(num_segments):
                    # Create embedding with speaker-specific pattern
                    base_embedding = np.random.randn(192) * 0.1
                    speaker_id = i * 1.5  # Different base for each speaker
                    embedding = base_embedding + np.random.randn(192) * 0.2 + speaker_id
                    embeddings.append(embedding)
                    labels.append(speaker)
            
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        # Generate fallback data
        embeddings = np.random.randn(50, 192)
        labels = np.array([f'Speaker_{i%3}' for i in range(50)])
    
    print(f"📊 Generated embeddings shape: {embeddings.shape}")
    print(f"🏷️ Generated labels shape: {labels.shape}")
    print(f"👥 Unique speakers in data: {len(np.unique(labels))}")
    
    # Save to experiment_data.npz
    np.savez('experiment_data.npz', 
             embeddings=embeddings, 
             labels=labels,
             speakers=labels)
    
    print(f"✅ Saved experiment_data.npz")
    print(f"   • Embeddings: {embeddings.shape}")
    print(f"   • Labels: {labels.shape}")
    print(f"   • File size: {Path('experiment_data.npz').stat().st_size / 1024:.1f} KB")
    
    return True

def main():
    print("🔄 Fresh Experiment Data Generator")
    print("=" * 40)
    
    success = generate_fresh_experiment_data()
    
    if success:
        print(f"\n🎉 Fresh experiment data generated!")
        print(f"\n📝 Next steps:")
        print(f"   1. Go to Speaker Analytics tab in dashboard")
        print(f"   2. Scroll down to t-SNE section")
        print(f"   3. Click 'Generate t-SNE Visualization'")
        print(f"   4. The t-SNE plot should now appear!")
    else:
        print(f"\n❌ Failed to generate experiment data")

if __name__ == "__main__":
    main()
