import torch
import torch.nn as nn

from torch.nn.functional import relu, sigmoid

class CNN(nn.Module):
    def __init__(self, in_channels: int = 1, kernel_size: int = 3, img_size: int = 200, dropout_rate: float = 0.2, radius_scale: int = 100) -> None:
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernel_size, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1)
        self.max_pool_1 = nn.MaxPool2d(2, 2)
        self.max_pool_2 = nn.MaxPool2d(2, 2)
        self.max_pool_3 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

        dim_1 = ((img_size - kernel_size + 2) + 1) // 2
        dim_2 = ((dim_1 - kernel_size + 2) + 1) // 2
        dim_3 = ((dim_2 - kernel_size + 2) + 1) // 2
        
        self.linear_1 = nn.Linear(in_features=128 * dim_3 * dim_3, out_features=512)
        self.linear_2 = nn.Linear(in_features=512, out_features=64)
        self.linear_3 = nn.Linear(in_features=64, out_features=3)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.img_size = img_size
        self.radius_scale = radius_scale

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)
        x = self.max_pool_1(x)


        x = self.conv2(x)
        x = self.bn2(x)
        x = relu(x)
        x = self.max_pool_2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = relu(x)
        x = self.max_pool_3(x)

        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = relu(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        x = relu(x)
        x = self.linear_3(x)
        return x

        # centers = self.img_size * sigmoid(x[:, :2]) 
        # radius = self.radius_scale * sigmoid(x[:, 2])  

        # return torch.cat([centers, radius.unsqueeze(1)], dim=1)