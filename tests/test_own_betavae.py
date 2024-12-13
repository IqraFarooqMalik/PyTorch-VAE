import os
import sys  # Import sys module
import argparse
import torch
import unittest
from models.beta_vae import BetaVAE
from dataset import VAEDataset
from experiment import VAEXperiment
from pytorch_lightning import Trainer
from torchsummary import summary
from pytorch_lightning.loggers import TensorBoardLogger
import yaml
import pytest

# Argument parser for the config file
parser = argparse.ArgumentParser(description='Test script for BetaVAE models')
parser.add_argument('--config', '-c', 
                    dest="filename",
                    metavar='FILE', 
                    help='Path to the config file', 
                    required=True)
args, unknown = parser.parse_known_args()  # Handle unrecognized arguments

# Load configuration from the provided file
with open(args.filename, 'r') as file:
    config = yaml.safe_load(file)

class TestBetaVAE(unittest.TestCase):

    def setUp(self) -> None:
        self.input_channels = 3
        self.latent_dim = 10
        self.loss_type = 'H'
        self.model = BetaVAE(self.input_channels, self.latent_dim, loss_type=self.loss_type).cuda()

        # Dynamically construct the checkpoint directory path from config
        tb_logger = TensorBoardLogger(
            save_dir=config['logging_params']['save_dir'],
            name=config['model_params']['name']
        )
        self.checkpoint_dir = os.path.join(tb_logger.log_dir, "checkpoints")
        print(f"Checkpoint directory: {self.checkpoint_dir}")

    def test_summary(self):
        """Test if the model's architecture is defined correctly."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)  # Move the model to the correct device
        print(summary(self.model, (3, 64, 64), device=device))


    def test_loss(self):
        """Test the loss function output."""
        x = torch.randn(16, 3, 64, 64).cuda()
        outputs = self.model(x)  # Get all outputs
        reconstructed, _, mu, log_var = outputs  # Unpack the required outputs

        # Example kld_weight for testing
        kld_weight = 0.01  # Adjust this value as needed for testing

        # Call the loss function
        loss_dict = self.model.loss_function(
            reconstructed, x, mu, log_var, M_N=kld_weight
        )

        # Print the loss components for debugging
        print("Loss:", loss_dict['loss'].item())
        print("Reconstruction Loss:", loss_dict['Reconstruction_Loss'].item())
        print("KLD:", loss_dict['KLD'].item())

        # Assertions to validate the loss outputs
        self.assertTrue('loss' in loss_dict)
        self.assertTrue('Reconstruction_Loss' in loss_dict)
        self.assertTrue('KLD' in loss_dict)
        self.assertGreater(loss_dict['loss'].item(), 0)
        self.assertGreater(loss_dict['Reconstruction_Loss'].item(), 0)
        self.assertGreater(loss_dict['KLD'].item(), 0)


    def test_forward(self):
        """Test the forward pass for output shapes."""
        x = torch.randn(16, 3, 64, 64).cuda()
        outputs = self.model(x)  # Capture all outputs
        reconstructed, _, mu, log_var = outputs  # Unpack outputs selectively

        # Print the shapes for debugging
        print("Reconstruction Output Shape:", reconstructed.size())
        print("Mean (mu) Output Shape:", mu.size())
        print("Log Variance (log_var) Output Shape:", log_var.size())

        # Add assertions
        self.assertEqual(reconstructed.shape, x.shape)
        self.assertEqual(mu.shape, (16, self.latent_dim))
        self.assertEqual(log_var.shape, (16, self.latent_dim))


    def test_dataset_forward(self):
        """Test model forward pass using the actual dataset."""

        data = VAEDataset(data_path=config["data_params"]["data_path"], transform=None, batch_size=16)  # Provide dataset path
        data.setup(stage='test')
        test_loader = data.test_dataloader()

        for batch in test_loader:
            x = batch['data'].cuda()  # Move to GPU
            reconstructed, latent, extra_output = self.model(x)  # Adjust unpacking
            print("Batch Output Shape:", reconstructed.shape)
            self.assertEqual(reconstructed.shape, x.shape)
            break

    def test_checkpoint_loading(self):
        """Test loading a model checkpoint."""
        checkpoint_path = self.checkpoint_dir  # Example path
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file does not exist at: {checkpoint_path}")
            self.fail("Checkpoint file is missing. Ensure the path is correct.")
        
        # If checkpoint exists, attempt to load it
        try:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully!")
        except Exception as e:
            self.fail(f"Error occurred while loading the checkpoint: {e}")

    def test_reconstruction(self):
        """Test the reconstruction output."""
        x = torch.randn(16, 3, 64, 64).cuda()
        outputs = self.model(x)  # Get all outputs
        reconstructed = outputs[0]  # Extract the reconstructed output
        
        # Print the reconstruction for debugging
        print("Reconstructed Output Shape:", reconstructed.size())
    
        # Assertions
        self.assertEqual(reconstructed.shape, x.shape, "Reconstruction shape should match input shape")

    def test_metrics(self):
        """Test performance metrics such as reconstruction loss and KL divergence."""
        x = torch.randn(16, 3, 64, 64).cuda()
        outputs = self.model(x)  # Get all outputs
        reconstructed = outputs[0]  # Reconstruction
        mu = outputs[2]  # Latent mean
        log_var = outputs[3]  # Latent log variance
        
        # Compute metrics
        recons_loss = torch.nn.functional.mse_loss(reconstructed, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        # Print metrics
        print("Reconstruction Loss:", recons_loss.item())
        print("KL Divergence Loss:", kld_loss.item())
        
        # Assertions
        self.assertTrue(recons_loss.item() > 0, "Reconstruction loss should be positive")
        self.assertTrue(kld_loss.item() >= 0, "KL divergence should be non-negative")


    def test_edge_cases(self):
        """Test edge cases for model stability."""
        # Create an empty tensor with shape [0, 3, 64, 64]
        empty_input = torch.empty(0, 3, 64, 64).cuda()

        # Check if a RuntimeError is raised
        with pytest.raises(RuntimeError):
            self.model(empty_input)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]] + unknown)  # Pass unrecognized arguments to unittest
