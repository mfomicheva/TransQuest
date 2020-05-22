import torch
import torch.nn as nn

from transformers.modeling_roberta import RobertaClassificationHead


class RobertaClassificationHeadInjection(RobertaClassificationHead):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHeadInjection, self).__init__(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.reduce = nn.Linear(config.hidden_size, 1)
        self.out_proj_reduced = nn.Linear(2, config.num_labels)

    def forward(self, features, model_score=None, **kwargs):
        batch_dim, max_len, hidd_dim = features.shape
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)  # shape: (B, H)
        assert x.shape == (batch_dim, hidd_dim)
        if model_score is not None:
            model_score = model_score.unsqueeze(1)  # shape: (B, H)
            assert model_score.shape == (batch_dim, 1)
            x = self.reduce(x)
            assert x.shape == (batch_dim, 1)
            x = torch.cat((x, model_score), 1)
            assert x.shape == (batch_dim, 2)
            x = self.out_proj_reduced(x)
        else:
            x = self.out_proj(x)
        return x
