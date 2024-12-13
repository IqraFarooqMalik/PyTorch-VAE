import torch
import unittest
from models.beta_vae import BetaVAE
from torchsummary import summary

class TestVAE(unittest.TestCase):

    def setUp(self) -> None:
        self.model = BetaVAE(3, 10, loss_type='H').cuda()  # Move model to GPU

    def test_summary(self):
        print(summary(self.model, (3, 64, 64), device='cuda'))  # Ensure summary uses GPU

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64).cuda()  # Move input to GPU
        y = self.model(x)
        print("Model Output size:", y[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64).cuda()  # Move input to GPU
        result = self.model(x)
        loss = self.model.loss_function(*result, M_N = 0.005)
        print(loss)


if __name__ == '__main__':
    unittest.main()
