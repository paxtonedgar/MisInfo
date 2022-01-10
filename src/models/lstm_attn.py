"""
Basic LSTM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.base_torch_model import BaseTorchModel


class LSTMAttn(BaseTorchModel):
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
        super(LSTMAttn, self).__init__(config)

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

        # embedding dropout
        self._embedding_dropout = nn.Dropout(net_conf['embedding_dropout'])

        # GRU
        self._gru = nn.LSTM(
            input_size=net_conf['embedding_size'],
            hidden_size=net_conf['hidden_size'],
            num_layers=net_conf['num_layers'],
            bias=net_conf['bias'],
            dropout=net_conf['gru_dropout'],
            bidirectional=net_conf['bidirectional']
        )

        num_directions = int(net_conf['bidirectional']) + 1

        # attention weights
        self._w_b_linear = nn.Linear(
            net_conf['hidden_size'] * num_directions, net_conf['hidden_size'],
            bias=True
        )
        self._context = nn.Parameter(torch.randn(net_conf['hidden_size'], 1))
        self._softmax = nn.Softmax(dim=0)

        # attention dropout
        self._attn_dropout = nn.Dropout(net_conf['attn_dropout'])

        # output
        self._output = nn.Linear(
            net_conf['hidden_size'] * num_directions, net_conf['num_classes']
        )

    def _batch_attn(self, x, x_len):
        sequence = None
        for i in range(x.shape[1]):
            attn_w = torch.mm(x[:, i, :], self._context)
            if sequence is None:
                sequence = attn_w
            else:
                sequence = torch.cat((sequence, attn_w), 1)
        max_len = x.shape[0]
        idx = torch.arange(max_len).unsqueeze(1).expand_as(sequence)
        len_expanded = x_len.unsqueeze(0).expand_as(sequence)
        mask = idx < len_expanded
        sequence[~mask] = float('-inf')
        return self._softmax(sequence)

    def forward(self, x, x_len):
        """
        Forward pass logic.
        """
        # x.shape: (sequence_lenght, batch_size)
        x = self._embedding(x)
        x = self._embedding_dropout(x)
        # x.shape: (sequence_length, batch_size, embedding_dim)
        packed_emb = pack_padded_sequence(x, x_len.tolist())
        # packed_emb.shape: (batch_sum_seq_len, embedding_dim)
        packed_out, _ = self._gru(packed_emb)
        # packed_out.shape: (batch_sum_seq_len, num_dir * hidden_size)
        rnn_out, len_out = pad_packed_sequence(packed_out)
        # rnn_out.shape: (sequence_length, batch_size, num_dir * hidden_size)
        w_b_out = self._w_b_linear(rnn_out)
        # w_b_out.shape: (sequence_length, batch_size, hidden_size)
        w_b_out = torch.tanh(w_b_out)
        # w_b_out.shape: (sequence_length, batch_size, hidden_size)
        attn_w = self._batch_attn(w_b_out, len_out)
        # attn_w.shape: (sequence_length, batch_size)
        alpha = attn_w.unsqueeze(2).expand_as(rnn_out)
        # alpha.shape: (sequence_length, batch_size, num_dir * hidden_size)
        x = alpha * rnn_out
        # x.shape: (sequence_length, batch_size, num_dir * hidden_size)
        x = torch.sum(x, dim=0)
        # x.shape: (batch_size, num_dir * hidden_size)
        logits = self._output(x)
        # logits.shape: (batch_size, num_classes)
        return logits, attn_w
