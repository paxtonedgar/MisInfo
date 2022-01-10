
import torch
import torchtext.data as data
import torchtext.vocab as vocab

from src.features.text_tokenizer import TextTokenizer


class DatasetWrapper(data.Dataset):
    """
    Custom wrapper for torchtext.data.Dataset class which is used for splitting
    data and passing data to pytorch.
    This class was inspired by an example provided by torchtext:
    https://github.com/pytorch/text/blob/master/torchtext/datasets/imdb.py
    """

    class Preprocessor(object):

        def __init__(self, max_tokens):
            self._max_tokens = max_tokens

        def preprocess(self, tokens):
            if self._max_tokens and self._max_tokens < len(tokens):
                return tokens[:self._max_tokens]
            return tokens

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text, labels, fields, indices, **kwargs):
        """
        :param text:
        :param labels:
        :param fields:
        :param indices:
        """

        # this is very slow, maybe there's a way to speed this up?
        examples = [
            data.Example.fromlist([
                text[int(idx)], text[int(idx)], labels[int(idx)]
            ], fields)
            for idx in indices
        ]

        super(DatasetWrapper, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text, labels, splits, fields, **kwargs):
        """
        Create dataset objects for splits of the dataset. The splits are created
        from lists of indices passed to the method.
        :param text:
        :param labels:
        :param splits:
        :param fields:
        :return:
        """
        train_data = cls(text, labels, fields, splits['train'], **kwargs)
        dev_data = cls(text, labels, fields, splits['dev'], **kwargs)
        test_data = cls(text, labels, fields, splits['test'], **kwargs)

        return train_data, dev_data, test_data

    @classmethod
    def iters(
            cls, text, labels, splits, batch_size=16, device=-1,
            embeddings_path=None, vector_cache=None, max_tokens=None, **kwargs
    ):
        """
        :param text:
        :param labels:
        :param splits:
        :param batch_size:
        :param device:
        :param embeddings_path:
        :param vector_cache:
        :param max_tokens:
        :return:
        """
        tt = TextTokenizer(stop=True, punct=True, lower=True, stem=False)
        pre = DatasetWrapper.Preprocessor(max_tokens)

        # define fields
        raw_field = data.RawField()
        text_field = data.Field(
            sequential=True, tokenize=tt.tokenize, preprocessing=pre.preprocess,
            include_lengths=True
        )
        label_field = data.Field(
            sequential=False, use_vocab=False, is_target=True
        )

        fields = [
            ('raw', raw_field), ('text', text_field), ('label', label_field)
        ]

        train_data, dev_data, test_data = cls.splits(
            text, labels, splits, fields, **kwargs
        )

        text_field.build_vocab(train_data, dev_data)

        # load custom pre-trained embeddings
        if embeddings_path:
            vectors = vocab.Vectors(embeddings_path, cache=vector_cache)
            text_field.vocab.set_vectors(
                vectors.stoi, vectors.vectors, vectors.dim
            )

        device = torch.device(
            'cpu' if device == -1 else 'cuda:{}'.format(device)
        )
        train_iter, dev_iter = data.BucketIterator.splits(
            (train_data, dev_data), batch_size=batch_size,
            shuffle=True, device=device, sort=False, sort_within_batch=True,
            repeat=False
        )
        test_iter = data.Iterator(
            test_data, batch_size=batch_size, sort=False, device=device,
            sort_within_batch=True, repeat=False
        )

        return train_iter, dev_iter, test_iter, text_field.vocab
