import torch
import torch.nn as nn

class CircleLoss(nn.Module):
    def __init__(self) -> None:
        super(CircleLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, output: torch.Tensor, target: torch.Tensor):
        assert output.size(1) == target.size(1) == 3, f"Output Size {output.size(2)} and Target Size {target.size(2)} does not match!"
        return self.mse_loss(output[:, :2], target[:, :2]) + self.l1_loss(output[:, 2], target[:, 2])
