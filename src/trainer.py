import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.utils import vae_sparse_loss

class VAETrainer:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.model(data)
            #loss = vae_loss_function(recon_batch, data, mu, logvar)
            # standard loss
            loss = vae_sparse_loss(recon_batch, data, mu, logvar)
            #sparse loss
            
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
            
        return total_loss / len(loader.dataset)