"""
Text tokenizer. Offers spacy-based, NLTK-based, and custom tokenization.
"""

import re
import string
from typing import List

import spacy
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class TextTokenizer():
    """
    Class for tokenizing input text. The class offers three different methods
    for tokenizing text -- one that uses standard NLTK tools, one that uses
    SpaCy, and one custom method taken from the following github repository:
    https://github.com/vipzhaoguangyao/cnn-text-classification-pytorch2
    """

    _stopwords = set(stopwords.words('english'))
    _punct_map = str.maketrans('', '', string.punctuation)
    _stemmer = PorterStemmer()
    _spacy_en = spacy.load('en_core_web_sm')

    def __init__(
            self, stop: bool = True, punct: bool = True, lower: bool = True,
            stem: bool = True, replace_nums: bool = False
    ) -> None:
        """
        Initialize class.

        :param stop: Remove stopwords? (True = remove, False = do not remove),
                     defaults to True
        :type stop: bool, optional
        :param punct: Remove punctuation? (True = remove, False = do not
                      remove), defaults to True
        :type punct: bool, optional
        :param lower: Covert words to lowercase? (True = convert),
                      defaults to True
        :type lower: bool, optional
        :param stem: Stem words? (True = stem, False = not stem),
                     defaults to True
        :type stem: bool, optional
        :param replace_nums: Replace numbers with a token? (True = replace),
                             defaults to False
        :type replace_nums: bool, optional
        :return: none
        :rtype: None
        """
        self._stop = stop
        self._stem = stem
        self._lower = lower
        self._punct = punct
        self._replace_nums = replace_nums

    @staticmethod
    def contains_alnum(text: str) -> bool:
        """
        Check if string contains at least one alphanumeric characters.

        :param text: text
        :type text: str
        :return: true if t contains at least one letter or number
        :rtype: bool
        """
        return any(c.isalnum() for c in text)

    @staticmethod
    def is_num(text: str) -> bool:
        """
        Check if string is a number (contains only digits).

        :param text: text
        :type text: str
        :return: true if t contains only digits
        :rtype: bool
        """
        return all(c.isdigit() for c in text)

    @staticmethod
    def get_stopwords() -> List[str]:
        """
        Get stopwords used by the class.

        :return: list of stopwords
        :rtype: List[str]
        """
        return TextTokenizer._stopwords

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using spacy tokenization.

        :param text: text to be processed
        :type text: str
        :return: list of tokens
        :rtype: List[str]
        """
        text = text.strip()
        # \x00 -- special character appearing in PDF files, represents NULL
        text = text.replace('\x00', ' ')
        # replace floating point numbers with a single token
        if self._replace_nums:
            text = re.sub(r"\d+\.\d+", ' floattoken ', text)
        tokens = []
        for tok in TextTokenizer._spacy_en.tokenizer(text):
            is_stop = tok.is_stop or tok.lower_ in TextTokenizer._stopwords
            if self._stop and is_stop:
                continue
            token = tok.lower_ if self._lower else tok.text
            if self._stem:
                token = TextTokenizer._stemmer.stem(token)
            if len(token) > 0:
                tokens.append(token)
        if self._replace_nums:
            tokens = [
                'inttoken' if TextTokenizer.is_num(t) else t for t in tokens
            ]
        if self._punct:
            tokens = [t.translate(TextTokenizer._punct_map) for t in tokens]
        tokens = [t.strip() for t in tokens]
        tokens = [
            t for t in tokens if TextTokenizer.contains_alnum(t) and len(t) > 1
        ]
        return tokens
