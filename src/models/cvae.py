import torch
import torch.nn as nn

class cVAE(nn.Module):

    def __init__(self, spatial_x_dim, out_dim):
        super(cVAE, self).__init__()

        self.encoderX = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=50, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=50, out_channels=25, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=25, out_channels=1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.3)
        )

        self.encoder = nn.Sequential(
            nn.Linear((spatial_x_dim * 1) + out_dim, 2000),
            nn.Tanh(),
            nn.Linear(2000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 500)
        )

        self.encoderY_mu = nn.Linear(500, 500)
        self.encoderY_logvar = nn.Linear(500, 500)

        self.decoder = nn.Sequential(
            nn.Linear((spatial_x_dim * 1) + 500, 1000),
            nn.Tanh(),
            nn.Linear(1000, 2000),
            nn.Tanh(),
            nn.Linear(2000, out_dim),
            nn.ReLU()
        )

    def encode(self, xy):
        zL = self.encoder(xy)
        return self.encoderY_mu(zL), self.encoderY_logvar(zL)

    def encodeX(self, x):
        return self.encoderX(x)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def get_latent_var(self, xy):
        mu, logvar = self.encode(xy)
        zy = self.reparametrize(mu, logvar)
        return zy

    def decode(self, z):
        y_prime = self.decoder(z)
        return y_prime

    def forward(self, x, y):
        Zx = self.encodeX(x)
        xy = torch.cat((Zx, y), dim=1)
        mu, logvar = self.encode(xy)
        zy = self.reparametrize(mu, logvar)
        z = torch.cat((Zx, zy), dim=1)
        y_prime = self.decode(z)
        return y_prime, mu, logvar

    def predictX(self, x):
        Zx = self.encodeX(x)
        latentDim = 500
        if torch.cuda.is_available():
            zy = torch.rand((Zx.shape[0], latentDim)).cuda()
        else:
            zy = torch.rand((Zx.shape[0], latentDim))
        z = torch.cat((Zx, zy), dim=1)
        y_prime = self.decode(z)
        return y_prime