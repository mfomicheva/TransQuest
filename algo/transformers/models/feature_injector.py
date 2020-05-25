import torch

from torch import nn


class FeatureInjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_features = config.num_features
        self.reduce = config.reduce

        self.reducer = nn.Linear(config.hidden_size, config.num_features)
        self.dense = nn.Linear(config.hidden_size + config.num_features, config.hidden_size + config.num_features)
        self.out_proj = nn.Linear(config.hidden_size + config.num_features, config.num_labels)

    def forward(self, x, features_inject):
        batch_dim, hidd_dim = x.shape
        if len(features_inject.shape) < 2:
            features_inject = features_inject.unsqueeze(1)  # shape: (B, H)
        assert features_inject.shape[1] == self.num_features
        if self.reduce:
            x = self.reducer(x)
            assert x.shape == (batch_dim, self.num_features)
        x = torch.cat((x, features_inject), dim=1)
        x = self.dense(x)
        x = self.out_proj(x)
        return x
