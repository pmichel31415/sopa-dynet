#!/usr/bin/env python3


class SentenceClassifier(object):
    """Interface for sentence classifier networks"""

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser"""
        pass

    @staticmethod
    def from_args(dic, num_classes, args):
        """Returns an instance of this model based on command line arguments"""
        raise NotImplementedError()

    def get_word_embeddings(self):
        """Return word embeddings lookup parameter"""
        raise NotImplementedError()
