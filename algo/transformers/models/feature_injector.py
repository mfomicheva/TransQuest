import torch

from torch import nn


class FeatureInjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_features = config.num_features
        self.reduce = config.reduce

        self.reducer = nn.Linear(config.hidden_size, 1)
        self.dense = nn.Linear(config.hidden_size + 1, config.hidden_size + 1)
        self.out_proj = nn.Linear(config.hidden_size + 1, config.num_labels)

    def forward(self, x, features_inject):
        batch_dim, hidd_dim = x.shape
        features = features_inject.unsqueeze(1)  # shape: (B, H)
        assert features.shape == (batch_dim, 1)
        if self.reduce:
            x = self.reducer(x)
            assert x.shape == (batch_dim, 1)
        x = torch.cat((x, features), 1)
        x = self.dense(x)
        x = self.out_proj(x)
        return x
