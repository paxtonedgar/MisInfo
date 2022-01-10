"""
Basic RNN.
"""

import torch
import torch.nn as nn

from src.models.base_torch_model import BaseTorchModel


class RNN(BaseTorchModel):
    """
    Basic RNN.
    """

    def __init__(self, config, vocab):
        """
        [summary]

        :param config: [description]
        :type config: [type]
        :param vocab: [description]
        :type vocab: [type]
        """
        super(RNN, self).__init__(config)
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
        self._rnn = nn.RNN(
            input_size=net_conf['embedding_size'],
            hidden_size=net_conf['hidden_size'],
            num_layers=net_conf['num_layers'],
            nonlinearity=net_conf['nonlinearity'],
            bias=net_conf['bias'],
            dropout=net_conf['dropout'],
            bidirectional=net_conf['bidirectional']
        )
        self._dropout = nn.Dropout(self._config['dropout'])
        num_directions = int(net_conf['bidirectional']) + 1
        self._output = nn.Linear(
            net_conf['hidden_size'] * net_conf['num_layers'] * num_directions,
            net_conf['num_classes']
        )

    def forward(self, x):
        """
        Forward pass logic.
        """
        # x.shape: (sequence_lenght, batch_size)
        x = self._embedding(x)
        # x.shape: (sequence_length, batch_size, embedding_dim)
        output, h_n = self._rnn(x)
        # h_n.shape = (num_layers * num_directions, batch_size, hidden_size)
        h_n = h_n.permute(1, 0, 2)
        # h_n.shape = (batch_size, num_layers * num_directions, hidden_size)
        h_n = h_n.contiguous().view(
            h_n.shape[0], h_n.shape[1] * h_n.shape[2]
        )
        # h_n.shape = (batch_size, num_layers * num_directions * hidden_size)
        logits = self._output(h_n)
        # logits.shape = (batch_size, num_classes)
        return logits
