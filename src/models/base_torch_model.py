"""
Base class for all NN models.
"""

import json
import logging
from typing import Dict, Any

import torch.nn as nn
import numpy as np


class BaseTorchModel(nn.Module):
    """
    Base class for all NN models.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the network.

        :param config: dictionary
        :type config: Dict[str, Any]
        :return: none
        :rtype: None
        """
        super(BaseTorchModel, self).__init__()
        self._logger = logging.getLogger(__name__)
        self._config = config

    @property
    def num_classes(self) -> int:
        return self._config['network']['num_classes']

    def forward(self, *input):
        """
        Forward pass logic.
        """
        raise NotImplementedError

    def summary(self) -> None:
        """
        Log model summary.

        :return: none
        :rtype: None
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self._logger.info('Trainable parameters: %s', params)
        self._logger.info(
            'Model configuration:\n%s', json.dumps(self._config, indent=4)
        )
        self._logger.info(self)
