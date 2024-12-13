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
        print(summary(self.model, (3, 64, 64), device='cpu'))

    def test_loss(self):
        """Test the loss function output."""
        x = torch.randn(16, 3, 64, 64).cuda()
        reconstructed, latent = self.model(x)
        loss = self.model.loss_function(reconstructed, x, latent, M_N=0.005)
        print("Loss Breakdown:", loss)
        self.assertIn('reconstruction_loss', loss)
        self.assertIn('kl_loss', loss)

    def test_forward(self):
        """Test the forward pass for output shapes."""
        x = torch.randn(16, 3, 64, 64).cuda()
        reconstructed, latent, extra_output = self.model(x)  # Adjust unpacking
        print("Reconstruction Output Shape:", reconstructed.size())
        print("Latent Output Shape:", latent.size())
        self.assertEqual(reconstructed.shape, x.shape)

    def test_dataset_forward(self):
        """Test model forward pass using the actual dataset."""
        data = VAEDataset(data_path="path_to_data", transform=None, batch_size=16)  # Provide dataset path
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
        checkpoint_path = os.path.join(self.checkpoint_dir, "last.ckpt")
        if os.path.exists(checkpoint_path):
            model = VAEXperiment.load_from_checkpoint(checkpoint_path)
            self.assertIsNotNone(model)
            print("Checkpoint successfully loaded")
        else:
            print(f"Checkpoint not found at: {checkpoint_path}")
            self.assertFalse(True)  # Fail the test if the checkpoint doesn't exist

    def test_reconstruction(self):
        """Test the reconstruction output."""
        x = torch.randn(16, 3, 64, 64).cuda()
        reconstructed, _ = self.model(x)
        self.assertEqual(reconstructed.shape, x.shape)
        print("Reconstruction Shape:", reconstructed.shape)

    def test_metrics(self):
        """Test performance metrics such as reconstruction loss and KL divergence."""
        x = torch.randn(16, 3, 64, 64).cuda()
        reconstructed, latent = self.model(x)
        loss_dict = self.model.loss_function(reconstructed, x, latent, M_N=0.005)
        print("Loss Breakdown:", loss_dict)
        self.assertGreater(loss_dict['reconstruction_loss'].item(), 0)
        self.assertGreater(loss_dict['kl_loss'].item(), 0)

    def test_edge_cases(self):
        """Test edge cases for model stability."""
        # Empty input tensor
        with self.assertRaises(RuntimeError):
            self.model(torch.empty(0, 3, 64, 64).cuda())
        
        # Small batch size
        x_small = torch.randn(1, 3, 64, 64).cuda()
        reconstructed, _ = self.model(x_small)
        self.assertEqual(reconstructed.shape, x_small.shape)

        # Large batch size
        x_large = torch.randn(64, 3, 64, 64).cuda()
        reconstructed, _ = self.model(x_large)
        self.assertEqual(reconstructed.shape, x_large.shape)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]] + unknown)  # Pass unrecognized arguments to unittest
