#!/usr/bin/env python3
import numpy as np


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

    @property
    def dic(self):
        """Return a dynet dictionary"""
        raise NotImplementedError()

    @property
    def word_embeddings(self):
        """Return word embeddings lookup parameter"""
        raise NotImplementedError()

    def load_pretrained_embeddings(self, filename, lowercase=False):
        """Load word embeddings from a glove-style file"""
        # Load vectors word by word
        with open(filename) as f:
            for line in f:
                word, vector_string = line.strip().split(" ", 1)
                if lowercase:
                    word = word.lower()
                if word in self.dic.indices:
                    wid = self.dic.index(word)
                    vector = np.fromstring(vector_string, sep=" ")
                    self.word_embeddings.init_row(wid, vector)

    def normalize_embeddings(self, norm=1.0):
        """Normalize word embeddings"""
        embed_matrix = self.word_embeddings.as_array()
        norms = np.linalg.norm(embed_matrix, axis=-1).reshape(-1, 1)
        norms = np.where(norms <= 0, 1e-20, norms)
        self.word_embeddings.init_from_array(embed_matrix / norms * norm)
