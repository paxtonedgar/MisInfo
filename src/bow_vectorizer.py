"""
Class for converting text to numerical vectors/matrices.
"""

import logging
from typing import Dict, Any, List, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.features.base_vectorizer import BaseVectorizer, DocumentMatrix


class BowVectorizer(BaseVectorizer):
    """
    Class for converting text to numerical vectors/matrices.
    """

    def __init__(
            self, tokenizer_cfg: Dict[str, Any],
            vectorizer_cfg: Dict[str, Any], tfidf: bool,
            log_level: int = logging.INFO
    ) -> None:
        """
        Initialize tokenizer and save embeddings path.

        :param tokenizer_cfg: config for tokenization
        :type tokenizer_cfg: Dict[str, Any]
        :param vectorizer_cfg: config for the TF-IDF vectorizer if used
        :type vectorizer_cfg: Dict[str, Any]
        :param tfidf: whether to use TFIDF weights
        :type tfidf: bool
        :param log_level: severity for logging, defaults to logging.INFO
        :type log_level: int, optional
        :return: none
        :rtype: None
        """
        super(BowVectorizer, self).__init__(
            tokenizer_cfg=tokenizer_cfg,
            vectorizer_cfg=vectorizer_cfg,
            tfidf=tfidf,
            log_level=log_level
        )
        self._logger = logging.getLogger(__name__)

    def vectorize(
            self, train_data: List[str], test_data: List[str] = None
    ) -> Tuple[DocumentMatrix, DocumentMatrix]:
        """
        Convert training and test documents into their vector representations.

        :return: tuple(numpy array of training vectors, numpy array of
                 test vectors), their sizes are [n_documents X size of selected
                 representation]
        :rtype: Tuple[DocumentMatrix, DocumentMatrix]
        """
        if test_data is None:
            self._logger.info('Vectorizing %d docs', len(train_data))
        else:
            self._logger.info(
                'Vectorizing %d/%d train/test docs',
                len(train_data), len(test_data)
            )
        vectorizer_cls = TfidfVectorizer if self._use_tfidf else CountVectorizer
        self._logger.info('Using %s', vectorizer_cls.__name__)
        vectorizer = vectorizer_cls(
            tokenizer=getattr(self._tt, 'tokenize'),
            **self._vectorizer_cfg
        )
        self._logger.debug('Fitting vectorizer to data')
        vectorizer.fit(train_data)
        self._logger.info('Transforming training & test data')
        if test_data is None:
            return vectorizer.transform(train_data)
        return (
            vectorizer.transform(train_data), vectorizer.transform(test_data)
        )
