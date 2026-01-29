import torch
import torch.nn as nn

class HematologyVAE(nn.Module):
    def __init__(self, latent_dim=128, input_channels=3):
        super(HematologyVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(input_channels, 32), 
            self._conv_block(32, 64),            
            self._conv_block(64, 128),         
            self._conv_block(128, 256),           
            nn.Flatten()
        )
        
        # Latent space distribution parameters
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            self._deconv_block(256, 128),         
            self._deconv_block(128, 64),          
            self._deconv_block(64, 32),           
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()                          
        )

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU()
        )

    def _deconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.decoder_input(z).view(-1, 256, 8, 8)), mu, logvar