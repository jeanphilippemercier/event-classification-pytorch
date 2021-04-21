from torch import nn
from torchvision import models

# Define the model


class CNN(nn.Module):
    # K represents the number of features
    def __init__(self, k):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
                      stride=2, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=2, padding_mode='zeros'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                      stride=2, padding_mode='zeros'),
            nn.ReLU()
        )
        # http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
        # "No zero padding, non-unit strides"
        # https://pytorch.org/docs/stable/nn.html
        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(28800, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, k)
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1)
        out = self.dense_layers(out)
        return out
