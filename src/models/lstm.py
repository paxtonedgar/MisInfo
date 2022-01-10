"""
Basic LSTM.
"""

import torch
import torch.nn as nn

from src.models.base_torch_model import BaseTorchModel


class LSTM(BaseTorchModel):
    """
    Basic LSTM.
    """

    def __init__(self, config, vocab) -> None:
        """
        [summary]

        :param config: [description]
        :type config: [type]
        :param vocab: [description]
        :type vocab: [type]
        """
        super(LSTM, self).__init__(config)

        self._config = config
        self._batch_size = config['dataloader_params']['batch_size']
        net_conf = self._config['network']

        # assign pre-trained embeddings
        if net_conf['pretrained_embeddings']:
            self._logger.info('Using pre-trained embeddings of size {}'.format(
                len(vocab)
            ))
            self._embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(vocab.vectors),
                freeze=net_conf['freeze_embeddings']
            )
        else:
            self._logger.info('Will train embeddings')
            self._embedding = nn.Embedding(
                len(vocab),
                net_conf['embedding_size']
            )
        self._lstm = nn.LSTM(
            input_size=net_conf['embedding_size'],
            hidden_size=net_conf['hidden_size'],
            num_layers=net_conf['num_layers'],
            bias=net_conf['bias'],
            dropout=net_conf['dropout'],
            bidirectional=net_conf['bidirectional']
        )
        self._dropout = nn.Dropout(self._config['dropout'])
        self._output = nn.Linear(
            net_conf['hidden_size'], net_conf['num_classes']
        )

    @property
    def num_classes(self) -> int:
        return self._config['network']['num_classes']

    def forward(self, x):
        """
        Forward pass logic.
        """
        # x.shape: (sequence_lenght, batch_size)
        x = self._embedding(x)
        x = self._dropout(x)
        # x.shape: (sequence_length, batch_size, embedding_dim)
        output, (h_n, c_n) = self._lstm(x)
        # predict based on last hidden state
        logits = self._output(h_n[-1])
        return logits
