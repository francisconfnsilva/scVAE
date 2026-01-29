# Sparse VAE for Hematological Label Assessment

This repository implements a Sparse Convolutional Variational Autoencoder (ScVAE) designed to detect label noise in bone marrow cell datasets.

## Methodology

### 1. Sparsity Regularization (L1 Norm)
Unlike standard VAEs, this model enforces sparsity in the latent representation using an L1 penalty on the latent mean ($\mu$).
* **Disentanglement:** Forces the model to use the fewest dimensions possible to describe a cell, mapping latent neurons to specific morphological features.

* **Noise Filtering:** Penalizes minor pixel variations to focus on high-level diagnostic features.

### 2. Label Assessment logic
The project uses a Suspicion Score by analyzing the entropy of neighboring classes in the latent space.
* **Suspicion Score:** Calculated by the mismatch rate between an image's assigned label and its $k$-nearest neighbors in the high-dimensional latent space.

## Project Structure
- `src/`: Core package containing the `HematologyVAE`, `VAETrainer`, and `utils`.
- `scripts/`: Entry points for data downloading and model execution.
- `data/`: Localized dataset storage.
- `results/`: Output directory for the `report.csv`.

## Getting Started

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt