import torch
import torch.nn as nn

from transformers.modeling_roberta import RobertaClassificationHead
from transquest.algo.transformers.models.feature_injector import FeatureInjector


class RobertaClassificationHeadInjection(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHeadInjection, self).__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.feature_injector = FeatureInjector(config)

    def forward(self, pretrained, reduce=False, features_inject=None, **kwargs):
        batch_dim, max_len, hidd_dim = pretrained.shape
        x = pretrained[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)  # shape: (B, H)
        assert x.shape == (batch_dim, hidd_dim)
        x = self.feature_injector(x, features_inject)
        return x
