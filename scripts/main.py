import sys
import os
import pandas as pd

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import HematologyVAE
from src.trainer import VAETrainer
from src.utils import get_assessment_data, quality_control
from src.data_loader import find_correct_root
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

def main():

    # 1. Setup Data
    data_path = find_correct_root("./data") 
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=data_path, transform=transform) 
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize Model & Trainer
    model = HematologyVAE(latent_dim=128) 
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = VAETrainer(model, optimizer) 
    
    # 3. Train
    print("Starting Sparse VAE Training...")
    for epoch in range(1, 26):
        loss = trainer.train_epoch(loader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.2f}")
    
    # 4. Assess Labels
    print("Training complete. Analyzing latent space for label noise...")
    latents, labels, paths = get_assessment_data(model, loader) 
    scores = quality_control(latents, labels) 
    
    # 5. Export Results
    os.makedirs("results", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    inv_map = {v: k for k, v in dataset.class_to_idx.items()}
    report_df = pd.DataFrame({
        'file_path': paths,
        'assigned_label': [inv_map[l] for l in labels],
        'suspicion_score': scores
    })
    
    report_df.sort_values('suspicion_score', ascending=False).to_csv('results/report.csv', index=False)
    torch.save(model.state_dict(), 'models/sparse_vae_final.pth')
    
    print(f"Report saved to results/report.csv")
    print(f"Model weights saved to models/sparse_vae_final.pth")

if __name__ == "__main__":
    main()