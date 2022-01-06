"""
Interface for vectorizers.
"""

import abc
import logging
from typing import Dict, Any, NewType, List, Union, Tuple

import gensim.downloader as api
from gensim.models import KeyedVectors

from src.features.text_tokenizer import TextTokenizer


CountMatrix = NewType('CountMatrix', 'numpy.ndarray[numpy.int64]')
WeightMatrix = NewType('WeightMatrix', 'numpy.ndarray[numpy.float64]')
DocumentMatrix = Union[CountMatrix, WeightMatrix]


class BaseVectorizer(object, metaclass=abc.ABCMeta):
    """
    Interface for vectorizers.
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
        :param vectorizer_cfg: config for the vectorizer
        :type vectorizer_cfg: Dict[str, Any]
        :param tfidf: whether to use TFIDF weights
        :type tfidf: bool
        :param log_level: severity for logging, defaults to logging.INFO
        :type log_level: int, optional
        :return: none
        :rtype: None
        """
        self._logger = logging.getLogger(__name__)
        self._tt = TextTokenizer(**tokenizer_cfg)
        self._vectorizer_cfg = vectorizer_cfg
        self._w2v = None
        self._use_tfidf = tfidf
        self._logger.setLevel(log_level)
        self._logger.info('Using TF-IDF: %s', self._use_tfidf)

    @property
    def tokenizer(self) -> TextTokenizer:
        """
        Get current tokenizer instance.

        :return: text tokenizer
        :rtype: TextTokenizer
        """
        return self._tt

    @property
    def w2v(self) -> KeyedVectors:
        if self._w2v is None:
            self._w2v = api.load("glove-twitter-25")
        return self._w2v

    @abc.abstractmethod
    def vectorize(
            self, train_data: List[str], test_data: List[str]
    ) -> Tuple[DocumentMatrix, DocumentMatrix]:
        """
        Convert training and test documents into their vector representations.

        :return: tuple(numpy array of training vectors, numpy array of
                 test vectors), their sizes are [n_documents X size of selected 
                 representation]
        :rtype: Tuple[DocumentMatrix, DocumentMatrix]
        """
        raise NotImplementedError('Must define `vectorize` to use this class')
