import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class VAEXperiment(pl.LightningModule):
    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = params.get('retain_first_backpass', False)
        
        # Store loss history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "reconstruction_loss": [],
            "kl_divergence": []
        }

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(
            *results,
            M_N=self.params['kld_weight'],  # KLD weight term
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        # Store loss values
        self.history["train_loss"].append(train_loss["loss"].item())
        self.history["reconstruction_loss"].append(train_loss["Reconstruction_Loss"].item())
        self.history["kl_divergence"].append(train_loss["KLD"].item())

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(
            *results,
            M_N=1.0,  # Weight for validation
            optimizer_idx=optimizer_idx,
            batch_idx=batch_idx
        )

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        # Store validation loss
        self.history["val_loss"].append(val_loss["loss"].item())

    def on_train_epoch_end(self):
        """Plot loss graphs after each epoch."""
        self.plot_loss_curves()

    # def plot_loss_curves(self):
    #     """Plots and saves training & validation loss curves."""
        
    #     plt.figure(figsize=(10, 6))

    #     # Define the scaling factor to match the range of loss and epoch values
    #     # epoch_scaling_factor = len(self.history["train_loss"]) / 10  # Scale epochs to fit the loss range
        
    #     # Training Loss
    #     epochs_train = [epoch / (len(self.history["train_loss"]) / 10) for epoch in range(1, len(self.history["train_loss"]) + 1)]
    #     plt.subplot(2, 2, 1)
    #     plt.plot(epochs_train, self.history["train_loss"], label="Training Loss", color="blue")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.legend()

    #     # Validation Loss
    #     epochs_val = [epoch / (len(self.history["val_loss"]) / 10) for epoch in range(1, len(self.history["val_loss"]) + 1)]
    #     plt.subplot(2, 2, 2)
    #     plt.plot(epochs_val, self.history["val_loss"], label="Validation Loss", color="orange")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.legend()

    #     # Reconstruction Loss
    #     epochs_reconstruction = [epoch / (len(self.history["reconstruction_loss"]) / 10) for epoch in range(1, len(self.history["reconstruction_loss"]) + 1)]
    #     plt.subplot(2, 2, 3)
    #     plt.plot(epochs_reconstruction, self.history["reconstruction_loss"], label="Reconstruction Loss", color="green")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.legend()

    #     # KL Divergence
    #     epochs_kl = [epoch / (len(self.history["kl_divergence"]) / 10) for epoch in range(1, len(self.history["kl_divergence"]) + 1)]
    #     plt.subplot(2, 2, 4)
    #     plt.plot(epochs_kl, self.history["kl_divergence"], label="KL Divergence", color="red")
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.legend()

    #     plt.tight_layout()
        
    #     # Save the figure
    #     save_path = os.path.join(self.logger.log_dir, f"graphs/loss_curves_Epoch_{self.current_epoch}.png")
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.savefig(save_path)
    #     plt.close()


# for resized
    def plot_loss_curves(self):
        """Plots and saves training & validation loss curves."""
        
        # Define total number of epochs (from 0 to 9)
        total_epochs = 10  # Assuming 10 epochs

        plt.figure(figsize=(10, 6))

        # Train Loss
        epochs_train = np.linspace(0, total_epochs-1, len(self.history["train_loss"]))
        plt.subplot(2, 2, 1)
        plt.plot(epochs_train, self.history["train_loss"], label="Training Loss", color="blue")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks(np.arange(0, total_epochs, 1))  # Set x-ticks to 0,1,2,...,9

        # Validation Loss
        epochs_val = np.linspace(0, total_epochs-1, len(self.history["val_loss"]))
        plt.subplot(2, 2, 2)
        plt.plot(epochs_val, self.history["val_loss"], label="Validation Loss", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks(np.arange(0, total_epochs, 1))  # Set x-ticks to 0,1,2,...,9

        # Reconstruction Loss
        epochs_reconstruction = np.linspace(0, total_epochs-1, len(self.history["reconstruction_loss"]))
        plt.subplot(2, 2, 3)
        plt.plot(epochs_reconstruction, self.history["reconstruction_loss"], label="Reconstruction Loss", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks(np.arange(0, total_epochs, 1))  # Set x-ticks to 0,1,2,...,9

        # KL Divergence
        epochs_kl = np.linspace(0, total_epochs-1, len(self.history["kl_divergence"]))
        plt.subplot(2, 2, 4)
        plt.plot(epochs_kl, self.history["kl_divergence"], label="KL Divergence", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks(np.arange(0, total_epochs, 1))  # Set x-ticks to 0,1,2,...,9

        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(self.logger.log_dir, f"graphs/loss_curves_Epoch_{self.current_epoch}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


        
    def on_validation_end(self) -> None:
        self.sample_images()
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        # Save original images
        vutils.save_image(
            test_input.data,
            os.path.join(
                self.logger.log_dir,
                "Originals",
                f"originals_{self.logger.name}_Epoch_{self.current_epoch}.png"
            ),
            normalize=True,
            nrow=12
        )

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           "Samples",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
