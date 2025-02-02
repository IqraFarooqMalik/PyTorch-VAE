import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class BetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.fc_mu = nn.Linear(hidden_dims[-1] * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 7 * 7, latent_dim)


        # Build Decoder
        modules = []

        # First layer: Linear projection from latent space to feature maps
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 7 * 7)  # Increased by 8x to ensure sufficient space for upsampling

        hidden_dims.reverse()

        # Upsampling layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                    hidden_dims[i + 1],
                                    kernel_size=4,  # Larger kernel for more upsampling
                                    stride=2,  # Doubling the spatial dimensions
                                    padding=1, 
                                    output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        # Final layer to produce the image with 3 channels (RGB)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], hidden_dims[-1],
                            kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], 3, kernel_size=3, padding=1),
            nn.Tanh()  # Ensures output is between [-1, 1]
        )

        # Make sure the output is resized to (224, 224)
        self.resize_layer = nn.Sequential(
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)  # After the convolutions
        # print("After convolutions:", result.shape)  # Print the shape of the result
        result = self.adaptive_pool(result)  # Apply adaptive pooling to get a 7x7 feature map
        # print("After pooling:", result.shape)  # Print the shape after pooling
        result = torch.flatten(result, start_dim=1) 

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 7, 7)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = self.resize_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        # Check the input size before passing through the encoder
        # print(f"Input size: {input.size()}")
        
        # Encoder step
        mu, log_var = self.encode(input)
        
        # Check the output size of the encoder (mu and log_var)
        # print(f"mu size: {mu.size()}")
        print(f"log_var size: {log_var.size()}")
        
        # Reparameterization trick to sample z from the latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Check the size of z
        print(f"z size (latent space): {z.size()}")
        
        # Decode the latent code z to reconstruct the image
        recons = self.decode(z)
        
        # Check the size of the reconstructed image
        print(f"Reconstructed size: {recons.size()}")
        
        # Return the output
        return [recons, input, mu, log_var]


    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
