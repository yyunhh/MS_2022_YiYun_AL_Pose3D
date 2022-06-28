import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class cVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, c):
        mu, log_var = self.encoder(x, c)
        z = self.reparameterize(mu, log_var)
        c = torch.flatten(c, start_dim=1)
        output = self.decoder(z,c)
        return output, mu, log_var
    
    def sample(self, num_samples, c, mean_zero= False):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param c: condition
        :return: (Tensor)
        """
        if mean_zero:
            z = torch.zeros(num_samples, 3) # number of samples, latent space dimension
        else:
            z = torch.randn(num_samples, 3) # number of samples, latent space dimension
        z = z.to(device)
        c = torch.flatten(c, start_dim=1)
        samples = self.decoder(z,c)
        samples = torch.reshape(samples, (num_samples, 17, 3)) #17
        return samples