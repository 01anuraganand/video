import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

def perform_tsne_pca_analysis(features, labels=None, output_dir="output/feature_analysis"):
    """
    Features: numpy array of shape (N, D) where N is num samples, D is feature dim.
    Labels: optional list or array of shape (N,)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # PCA
    print("Performing PCA...")
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(features)
    
    plt.figure(figsize=(8,6))
    if labels is not None:
        scatter = plt.scatter(pca_results[:,0], pca_results[:,1], c=labels, cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(pca_results[:,0], pca_results[:,1], alpha=0.7)
    plt.title('PCA of Model Features')
    plt.savefig(os.path.join(output_dir, "pca_analysis.png"))
    plt.close()

    # t-SNE
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, len(features)//3)), random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(8,6))
    if labels is not None:
        scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels, cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
    else:
        plt.scatter(tsne_results[:,0], tsne_results[:,1], alpha=0.7)
    plt.title('t-SNE of Model Features')
    plt.savefig(os.path.join(output_dir, "tsne_analysis.png"))
    plt.close()
    
    print(f"Feature analysis saved to {output_dir}")
