import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
import argparse
import os

# Set academic style for plots (Times New Roman-ish, clear lines)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300  # Q1 Standard Resolution
})

def generate_paper_visualization(embeddings, labels, output_filename="figure_1_tsne.png"):
    """
    Generates a Q1-standard t-SNE scatter plot.
    
    Args:
        embeddings: List or Numpy array of TitaNet embeddings (Shape: N x 192)
        labels: List of strings (e.g., ["Male_Real", "Female_Clone", ...])
        output_filename: Path to save the high-res image
    """
    print(f"Computing t-SNE projection for {len(embeddings)} embeddings...")
    
    if len(embeddings) < 2:
        print("Not enough embeddings to generate t-SNE plot.")
        return

    # 1. Pre-process with PCA (Standard practice to reduce noise before t-SNE)
    n_components = min(50, len(embeddings))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(embeddings)
    
    # 2. Run t-SNE
    # perplexity: Balance between local vs global structure. 30 is standard.
    perplexity = min(30, max(5, len(embeddings) // 3))
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, max_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(pca_result)
    
    # 3. Prepare Data for Plotting
    df = pd.DataFrame()
    df['t-SNE Dimension 1'] = tsne_results[:,0]
    df['t-SNE Dimension 2'] = tsne_results[:,1]
    df['Identity'] = labels
    
    # 4. Create the Plot
    plt.figure(figsize=(10, 8))
    
    # Define a high-contrast color palette suitable for colorblind readers (Academic standard)
    sns.scatterplot(
        x="t-SNE Dimension 1", y="t-SNE Dimension 2",
        hue="Identity",
        style="Identity", # Different shapes for different classes
        palette="bright",
        data=df,
        s=100, # Marker size
        alpha=0.8, # Slight transparency
        edgecolor='w', # White border around dots
        linewidth=0.5
    )
    
    plt.title("T-SNE Visualization of Speaker Embeddings (TitaNet-L)", pad=20, weight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # 5. Save
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-ready t-SNE plots from speaker embeddings")
    parser.add_argument("--input", default="experiment_data.npz", help="Path to saved experiment data (.npz)")
    parser.add_argument("--output", default="speaker_tsne_plot.png", help="Path to save the output plot")
    parser.add_argument("--test", action="store_true", help="Run with mock data for testing")
    
    args = parser.parse_args()
    
    if args.test:
        print("Generating mock data...")
        real_male = np.random.normal(loc=0.5, scale=0.1, size=(50, 192))
        clone_female = np.random.normal(loc=0.55, scale=0.1, size=(50, 192))
        noise = np.random.normal(loc=-0.5, scale=0.2, size=(30, 192))
        X = np.vstack([real_male, clone_female, noise])
        y = ["Male (Real)"] * 50 + ["Female (Clone)"] * 50 + ["Noise"] * 30
        generate_paper_visualization(X, y, args.output)
    elif os.path.exists(args.input):
        data = np.load(args.input)
        generate_paper_visualization(data['embeddings'], data['labels'], args.output)
    else:
        print(f"Error: Input file {args.input} not found. Run analysis with diarization first or use --test.")
