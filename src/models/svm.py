"""
SVM
"""

import logging
from typing import Dict, Union, Any

import numpy as np
from sklearn.svm import SVC

from src.evaluation.metrics import Metrics
from src.models.base_model import BaseModel, DataMatrix, LabelVector


class SVM(SVC, BaseModel):
    """
    SVM
    """

    def __init__(self, model_cfg: Dict[str, Any] = None) -> None:
        """Initialize class, pass config to parent

        :param model_cfg: model config
        :type model_cfg: Dict[str, Any], default None
        :return: None
        :rtype: None
        """
        if not model_cfg:
            model_cfg = {}
        super(SVM, self).__init__(**model_cfg)
        self._logger = logging.getLogger(__name__)

    def train_test(
            self, X_train: DataMatrix, X_test: DataMatrix,
            y_train: LabelVector, y_test: LabelVector,
            y_test_gt: LabelVector = None
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
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        labels = list(set(y_train).union(set(y_test)))

        # fit
        self._logger.info('Fitting %s classifier to data', __name__)
        self.fit(X_train, y_train)

        # predict
        self._logger.info('Done fitting to data, obtaining predictions')
        pred_train = self.predict(X_train)
        pred_test = self.predict(X_test)
        results = {
            f'train_{k}': v for k, v in
            Metrics.metrics(pred_train, y_train, labels).items()
        }
        results.update({
            f'test_{k}': v for k, v in
            Metrics.metrics(pred_test, y_test, labels).items()
        })
        if y_test_gt is not None:
            results.update({
                f'gt_{k}': v for k, v in
                Metrics.metrics(pred_test, y_test_gt, labels).items()
            })
        self._logger.info('Done testing %s', __name__)
        return results
