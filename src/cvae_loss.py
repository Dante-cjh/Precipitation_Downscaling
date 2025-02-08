import torch
import torch.nn.functional as F

def lossVAE(y, y_prime, mu, logvar, epoch):
    '''
    Standard VAE Loss (Reconstruction + KL) using L2 loss function (MSE)
    '''
    KL_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    MSE_loss = F.mse_loss(y, y_prime)
    y_loss = KL_loss + MSE_loss
    return y_loss