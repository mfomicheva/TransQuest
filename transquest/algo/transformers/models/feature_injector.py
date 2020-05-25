import torch

from torch import nn


class FeatureInjector(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.combinator = Reduce(config) if config.reduce else Concat(config)

    def forward(self, x, features_inject):
        return self.combinator(x, features_inject)


class Combinator(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_features = config.num_features
        self.num_labels = config.num_labels
        self.hidden_dim = config.hidden_size

    @staticmethod
    def prepare_features_inject(features_inject):
        if len(features_inject.shape) < 2:
            features_inject = features_inject.unsqueeze(1)  # shape: (B, H)
        return features_inject


class Reduce(Combinator):

    def __init__(self, config):
        super(Reduce, self).__init__(config)
        self.reducer = nn.Linear(self.hidden_dim, self.num_features)
        self.dense = nn.Linear(self.num_features * 2, self.num_features * 2)
        self.out_proj = nn.Linear(self.num_features * 2, self.num_labels)

    def forward(self, x, features_inject):
        batch_dim, hidd_dim = x.shape
        features_inject = self.prepare_features_inject(features_inject)
        assert features_inject.shape[1] == self.num_features
        x = self.reducer(x)
        assert x.shape == (batch_dim, self.num_features)
        x = torch.cat((x, features_inject), dim=1)
        x = self.dense(x)
        x = self.out_proj(x)
        return x


class Concat(Combinator):

    def __init__(self, config):
        super(Concat, self).__init__(config)
        self.dense = nn.Linear(self.hidden_dim + self.num_features, self.hidden_dim + self.num_features)
        self.out_proj = nn.Linear(self.hidden_dim + self.num_features, self.num_labels)

    def forward(self, x, features_inject):
        features_inject = self.prepare_features_inject(features_inject)
        assert features_inject.shape[1] == self.num_features
        x = torch.cat((x, features_inject), dim=1)
        x = self.dense(x)
        x = self.out_proj(x)
        return x
