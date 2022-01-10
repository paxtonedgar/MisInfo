"""
Experiments metrics.
"""

from typing import Dict, Union, NewType, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support
)


LabelArray = NewType('LabelArray', 'numpy.ndarray[numpy.int64]')


class Metrics(object):
    """
    Experiments metrics.
    """

    @staticmethod
    def metrics(
            pred: LabelArray, y: LabelArray, labels: List[Any]
    ) -> Dict[str, Union[str, float]]:
        """
        Calculate metrics from predictions and true labels.

        :param pred: predictions
        :type pred: LabelArray
        :param y: true labels
        :type y: LabelArray
        :return: Dict[str, Union[str, float]]
        """
        y = np.array(y)
        pred = np.array(pred)

        if len(pred) < 1:
            return {
                'n_samples': 0,
                'pos_samples': 0,
                'neg_samples': 0,
                'correct': 0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f_score': 0.0,
                'tn': 0,
                'tp': 0,
                'fn': 0,
                'fp': 0
            }
        precision, recall, f_score, _ = precision_recall_fscore_support(
            y, pred, labels=labels, average='macro', zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=labels).ravel()
        return {
            'n_samples': int(len(y)),
            'pos_samples': int(sum(y)),
            'neg_samples': int(len(y) - sum(y)),
            'correct': int(sum(y == pred)),
            'accuracy': float(accuracy_score(y, pred)),
            'precision': float(precision),
            'recall': float(recall),
            'f_score': float(f_score),
            'tn': int(tn),
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp)
        }

    @staticmethod
    def average_results(
            split_results: Dict[str, Dict[str, Union[int, float]]]
    ) -> Dict[str, float]:
        """
        Calculate average values over all splits in split_results.

        :param split_results: dictionary with individual split results
        :type split_results: Dict[str, Dict[str, Union[int, float]]]
        :return: dictionary with a single key: split_avg
        :rtype: Dict[str, float]
        """
        results = pd.DataFrame.from_dict(split_results, orient='index')
        return {'split_avg': results.mean(axis=0).to_dict()}
