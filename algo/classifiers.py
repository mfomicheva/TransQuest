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
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        if model_score is not None:
            model_score = model_score.unsqueeze(0)
            x = self.reduce(x)
            x = torch.cat((x, model_score), 0)
            x = x.transpose(1, 0)
            x = self.out_proj_reduced(x)
        else:
            x = self.out_proj(x)
        return x
