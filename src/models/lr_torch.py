"""
Linear Regression in PyTorch.
"""

from typing import Dict, Any

import torch
import torch.nn as nn
import torchtext

from src.models.base_torch_model import BaseTorchModel


class LRTorch(BaseTorchModel):
    """
    Linear Regression in PyTorch.
    """

    def __init__(
            self, config: Dict[str, Any], vocab: torchtext.vocab.Vocab
    ) -> None:
        """
        Initialize the network.

        :param config: dictionary
        :type config: Dict[str, Any]
        :param vocab: obtained from torchtext.data.Field.vocab built using
                      torchtext.data.Field.build_vocab()
        :return: none
        :rtype: None
        """
        super(LRTorch, self).__init__(config)
        self._config = config

        net_conf = self._config['network']
        self._logger.info('Vocabulary size: %d', len(vocab))

        # assign pre-trained embeddings
        if net_conf['pretrained_embeddings']:
            self._logger.info('Using pre-trained embeddings')
            self._embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(vocab.vectors),
                freeze=net_conf['freeze_embeddings']
            )
        else:
            self._logger.info('Will train embeddings')
            self._embedding = nn.Embedding(
                len(vocab), net_conf['embedding_size']
            )

        self.linear = nn.Linear(net_conf['input_dim'], net_conf['num_classes'])

    def forward(self, x):
        """
        Forward pass logic.
        :return: None
        """
        # x.shape: (max_tweet_length, batch_size)
        x = self._embedding(x)
        # x.shape: (max_tweet_length, batch_size, embedding_dim)
        x = torch.mean(x, dim=0, keepdim=False)
        # x.shape: (batch_size, embedding_dim)
        outputs = torch.sigmoid(self.linear(x))
        # outputs.shape:
        return outputs
