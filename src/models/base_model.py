"""
Interface for baseline models.
"""

import os
import abc
import json
from typing import Dict, Union, Iterable, NewType, Any

import numpy as np


DataMatrix = NewType('DataMatrix', np.ndarray)
LabelVector = NewType('LabelVector', Iterable[int])



class BaseModel(object, metaclass=abc.ABCMeta):
    """
    Interface for baseline models.
    """

    def _save_config(self, save_directory: str, config: Dict[str, Any]) -> None:
        """
        Save model config.

        :param save_directory: where to save the config
        :type: save_directory: str
        :param config: model config
        :type config: Dict[str, Any]
        :return: none
        :rtype: None
        """
        save_path = os.path.join(save_directory, 'model_config.json')
        with open(save_path, 'w') as fp_out:
            json.dump(config, fp_out, indent=4, sort_keys=False)

    @abc.abstractmethod
    def train_test(
            self, X_train: DataMatrix, X_test: DataMatrix,
            y_train: LabelVector, y_test: LabelVector
    ) -> Dict[str, Union[int, float]]:
        """Train a dummy classifier and return predictions on test data.

        :param X_train: numpy array of shape [n_train_samples, n_features]
        :type X_train: DataMatrix
        :param X_test: numpy array of shape [n_test_samples, n_features]
        :type X_test: DataMatrix
        :param y_train: numpy array of shape [n_train_samples]
        :type y_train: LabelVector
        :param y_test: numpy array of shape [n_test_samples]
        :type y_test: LabelVector
        :return: performance metrics
        :rtype: Dict[str, Union[int, float]]
        """
        raise NotImplementedError('Must define `train_test` to use this class')
