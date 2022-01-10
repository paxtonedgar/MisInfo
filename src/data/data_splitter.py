"""
Class for splitting data into train, dev, and test sets and for splitting
for cross-validation
"""

import math
import logging
from typing import Tuple, Dict, List, Iterable

import numpy as np
from sklearn.model_selection import (
    ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold, KFold
)


class DataSplitter():
    """
    Class for splitting data into train, dev, and test sets and for splitting
    for cross-validation
    """

    def __init__(
            self, dev_ratio: float = 0.15, test_ratio: float = 0.2,
            n_splits: int = None, random_state: int = None, **kwargs
    ) -> None:
        """
        Initialize class. If n_splits is set, it will override test_ratio.

        :param dev_ratio: proportion of data to be withheld for model
                          development, defaults to 0.15
        :type dev_ratio: float, optional
        :param test_ratio: proportion of data to be withheld for testing,
                           defaults to 0.2
        :type test_ratio: float, optional
        :param n_splits: number of splits in case of splitting for
                         cross-validation, defaults to None
        :type n_splits: int, optional
        :param random_state: the seed used by the random number generator,
                             defaults to None
        :type random_state: int, optional
        :return: none
        :rtype: None
        """
        self._logger = logging.getLogger(__name__)
        self._dev_ratio = dev_ratio
        self._test_ratio = test_ratio if not n_splits else 1.0 / n_splits
        self._n_splits = n_splits if n_splits else 1
        self._random_state = random_state
        self._logger.info('Using random state: %d', self._random_state)
        self._logger.info('Generating %d splits', self._n_splits)
        if self._dev_ratio is not None:
            self._logger.info(
                'Splits: %.2f/%.2f/%.2f train/dev/test',
                1.0 - self._dev_ratio - self._test_ratio,
                self._dev_ratio, self._test_ratio
            )
        else:
            self._logger.info(
                'Splits: %.2f/%.2f train/test',
                1.0 - self._test_ratio, self._test_ratio
            )

    @property
    def n_splits(self) -> int:
        """
        Number of splits to be produced.

        :return: number of splits
        :rtype: int
        """
        return self._n_splits

    def _shuffle_split(
            self, X: 'np.ndarray[Any]', y: 'np.ndarray[Any]', test_size: int,
            stratified: bool = True
    ) -> Iterable[Tuple['np.ndarray[int]', 'np.ndarray[int]']]:
        """
        Yield indices to split data into training and test sets.

        :param X: matrix with data
        :type X: np.ndarray[Any]
        :param y: array with labels
        :type y: np.ndarray[Any]
        :param test_size: required size of test set
        :type test_size: int
        :param stratified: preserve label distribution?, defaults to True
        :type stratified: bool, optional
        :return: iterable over tuples of [x_indices, y_indices]
        :rtype: Iterable[Tuple[np.ndarray[int], np.ndarray[int]]]
        """
        split_class = (
            StratifiedShuffleSplit
            if stratified and y is not None else ShuffleSplit
        )
        split = split_class(
            n_splits=self._n_splits,
            train_size=None,
            test_size=test_size,
            random_state=self._random_state
        )
        self._logger.info(
            'Generating splits with %s', split.__class__.__name__
        )
        return split.split(X, y)

    def _k_fold_split(
            self, X: 'np.ndarray[Any]', y: 'np.ndarray[Any]',
            stratified: bool = True
    ) -> Iterable[Tuple['np.ndarray[int]', 'np.ndarray[int]']]:
        """
        Yield indices to split data into training and test sets.

        :param X: matrix with data
        :type X: np.ndarray[Any]
        :param y: array with labels
        :type y: np.ndarray[Any]
        :param stratified: preserve label distribution?, defaults to True
        :type stratified: bool, optional
        :return: iterable over tuples of [x_indices, y_indices]
        :rtype: Iterable[Tuple[np.ndarray[int], np.ndarray[int]]]
        """
        split_class = StratifiedKFold if stratified and y is not None else KFold
        split = split_class(
            n_splits=self._n_splits,
            shuffle=True,
            random_state=self._random_state
        )
        self._logger.info(
            'Generating splits with %s', split.__class__.__name__
        )
        return split.split(X, y)

    def get_split_ids(
            self, X: 'np.ndarray[Any]', y: 'np.ndarray[Any]' = None,
            stratified: bool = True
    ) -> Iterable[Dict[str, List[int]]]:
        """
        Yield an iterator over (1 or n_splits) splits.

        :param X: matrix with data
        :type X: np.ndarray[Any]
        :param y: array with labels
        :type y: np.ndarray[Any]
        :param stratified: preserve label distribution?, defaults to True
        :type stratified: bool, optional
        :return: iterable over dictionaries of {'indices_key': list[int]}
        :rtype: Iterable[Dict[str, List[int]]]
        """
        # prepare data
        X = np.array(X)
        y = np.array(y) if y is not None else None
        n_samples = X.shape[0]
        i = np.arange(0, n_samples)
        self._logger.info(
            'Samples: %d, labels size: %s, idx size: %s',
            n_samples, y.shape if y is not None else 0, i.shape
        )

        # how many samples should go into train, dev and test
        dev_size = (
            math.ceil(n_samples * self._dev_ratio)
            if self._dev_ratio is not None else 0
        )
        test_size = math.ceil(n_samples * self._test_ratio)
        train_size = n_samples - dev_size - test_size
        self._logger.info(
            'Train, dev, test sizes: %d/%d/%d',
            train_size, dev_size, test_size
        )

        # split data into train+dev and test
        split_iter = (
            self._k_fold_split(X, y, stratified) if self._n_splits >= 2
            else self._shuffle_split(X, y, test_size, stratified)
        )
        for split in split_iter:
            train_dev_indices, test_indices = split[0], split[1]
            self._logger.info(
                'Train+dev size: %s, test size: %s',
                train_dev_indices.shape, test_indices.shape
            )

            # split train+dev into train and dev
            if self._dev_ratio is not None:
                self._logger.info('Splitting train into train and dev sets.')
                train_indices, dev_indices = next(self._shuffle_split(
                    X[train_dev_indices],
                    y[train_dev_indices] if y is not None else None,
                    dev_size, stratified
                ))
            else:
                train_indices = train_dev_indices
                dev_indices = None

            # return indices
            ii = i[train_dev_indices]
            # returning a dictionary because it's easier to pass it around
            yield {
                'train': ii[train_indices] if dev_indices is not None else ii,
                'dev': ii[dev_indices] if dev_indices is not None else [],
                'test': i[test_indices]
            }
