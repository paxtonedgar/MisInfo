
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_torch_model import BaseTorchModel


class CNN(BaseTorchModel):

    def __init__(self, config, vocab):
        """Initialize the network
        :param config:
        :param vocab: obtained from torchtext.data.Field.vocab built using
                      torchtext.data.Field.build_vocab()

        """
        super(CNN, self).__init__(config)
        self._config = config

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
        self._convolutions = nn.ModuleList([
            nn.Conv1d(
                in_channels=net_conf['embedding_size'],
                out_channels=net_conf['num_kernels'],
                kernel_size=k
            )
            for k in net_conf['kernel_sizes']
        ])
        self._dropout = nn.Dropout(self._config['dropout'])
        self._output = nn.Linear(
            net_conf['num_kernels'] * len(net_conf['kernel_sizes']),
            net_conf['num_classes']
        )

    def forward(self, x):
        """Forward pass logic.
        :return: None
        """
        # x.shape: (sequence_lenght, batch_size)
        x = self._embedding(x)
        # x.shape: (sequence_length, batch_size, embedding_dim)
        x = x.permute(1, 2, 0)
        # x.shape: (batch_size, embedding_dim, sequence_length)
        x = [F.relu(c(x)) for c in self._convolutions]
        # x.shape: (batch_size, num_kernels, sequence_length)
        # for each kernel size
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # x.shape: (batch_size, num_kernels)
        # for each kernel size
        x = torch.cat(x, 1)
        # x.shape: (batch_size, num_kernels * len(kernel_sizes))
        x = self._dropout(x)
        # x.shape: (batch_size, num_kernels * len(kernel_sizes))
        logit = self._output(x)
        # logit.shape: (batch_size, num_classes)
        return logit
