import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

def plot_samples(df, num_samples=5):
    plt.figure(figsize=(15, 5))
    samples = df.sample(num_samples)
    
    for i, (idx, row) in enumerate(samples.iterrows()):
        img = plt.imread(row['path'])
        plt.subplot(1, num_samples, i+1)
        plt.imshow(img)
        plt.title(row['label'])
        plt.axis('off')
    plt.show()

def plot_anomalies(file_paths, labels, scores, class_map, n=5):
    top_indices = np.argsort(scores)[::-1][:n]
    inv_class_map = {v: k for k, v in class_map.items()}

    plt.figure(figsize=(20, 4))
    for i, idx in enumerate(top_indices):
        img = plt.imread(file_paths[idx])
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.title(f"Label: {inv_class_map[labels[idx]]}\nSuspicion: {scores[idx]:.2f}")
        plt.axis('off')
    plt.show()

def generate_cells(model, latent_dim, device, num_samples=8):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decoder(model.decoder_input(z).view(-1, 256, 8, 8))
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            img = samples[i].cpu().permute(1, 2, 0).numpy()
            plt.subplot(1, num_samples, i+1)
            plt.imshow(img)
            plt.axis('off')
        plt.show()

def get_assessment_data(model, loader, device='cuda'):
    model.eval()
    all_mu = []
    all_labels = []
    all_paths = [path for path, label in loader.dataset.samples]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            encoded = model.encoder(images)
            mu = model.fc_mu(encoded)
            
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.vstack(all_mu), np.concatenate(all_labels), all_paths

def quality_control(latents, labels, k=10):
    # Find the k-nearest neighbors for every point in the latent space
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(latents)
    distances, indices = nbrs.kneighbors(latents)
    
    suspicion_scores = []
    
    for i in range(len(labels)):
        # Get labels of the neighbors
        neighbor_labels = labels[indices[i][1:]]
        # Calculate what percentage of neighbors have a DIFFERENT label
        mismatch_rate = np.mean(neighbor_labels != labels[i])
        suspicion_scores.append(mismatch_rate)
        
    return np.array(suspicion_scores)

def vae_sparse_loss(recon_x, x, mu, logvar, beta=1.0, l1_lambda=1e-4):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    l1_loss = torch.norm(mu, p=1) 
    return recon_loss + (beta * kld_loss) + (l1_lambda * l1_loss)

def plot_tsne(latents, labels, class_map):
    
    print("Running t-SNE")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    latents_2d = tsne.fit_transform(latents)
    
    inv_class_map = {v: k for k, v in class_map.items()}
    df_tsne = pd.DataFrame({
        'x': latents_2d[:, 0],
        'y': latents_2d[:, 1],
        'Cell Type': [inv_class_map[l] for l in labels]
    })
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_tsne, 
        x='x', y='y', 
        hue='Cell Type', 
        palette='hls', 
        alpha=0.6,
        s=50
    )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.tight_layout()
    plt.show()