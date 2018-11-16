#!/usr/bin/env python3

import dynet as dy

from dynn.layers import Affine
from dynn.layers import Embeddings
from dynn.layers import MeanPool1D
from dynn.layers import LSTM
from dynn.layers import Transduction, Bidirectional

from dynn.data.batching import SequenceBatch

from .base_classifier import SentenceClassifier

# Define the network as a custom layer


class BiLSTM(SentenceClassifier):
    """Your friendly neighborhood BiLSTM"""

    def __init__(self, dic, dx, dh, nc, dropout):
        # Master parameter collection
        self.pc = dy.ParameterCollection()
        # Word embeddings
        self.embed = Transduction(
            Embeddings(self.pc, dic, dx, pad_mask=0.0)
        )
        # BiLSTM
        self.bilstm = Bidirectional(
            forward_cell=LSTM(self.pc, dx, dh, dropout, dropout),
            backward_cell=LSTM(self.pc, dx, dh, dropout, dropout),
            output_only=True,
        )
        # Pooling layer
        self.pool = MeanPool1D()
        # Softmax layer
        self.softmax = Affine(self.pc, dh, nc, dropout=dropout)

    @staticmethod
    def add_args(parser):
        sopa_group = parser.add_argument_group("BiLSTM specific arguments")
        sopa_group.add_argument("--embed-dim", default=100,
                                help="Word embedding dim", type=int)
        sopa_group.add_argument("--hidden-dim", default=100,
                                help="Hidden dim", type=int)
        sopa_group.add_argument("--dropout", default=0.1,
                                help="Dropout", type=float)

    @staticmethod
    def from_args(dic, num_classes, args):
        return BiLSTM(
            dic,
            args.embed_dim,
            args.hidden_dim,
            num_classes,
            args.dropout,
        )

    def get_word_embeddings(self):
        """Return word embeddings lookup parameter"""
        return self.embed.layer.params

    def init(self, test=False, update=True):
        self.embed.init(test=test, update=update)
        self.bilstm.init(test=test, update=update)
        self.pool.init(test=test, update=update)
        self.softmax.init(test=test, update=update)

    def __call__(self, batch, return_embeds=False):
        # Handle simple sequence
        if not isinstance(batch, SequenceBatch):
            return self.__call__(SequenceBatch(batch), return_embeds)
        # Embed the f out of the inputs
        w_embeds = self.embed(batch.sequences)
        # Run the bilstm
        fwd_H, bwd_H = self.bilstm(w_embeds, lengths=batch.lengths)
        H = [0.5 * (fh + bh) for fh, bh in zip(fwd_H, bwd_H)]
        # Mask and stack to a matrix
        pooled_H = self.pool(H, lengths=batch.lengths)
        # Maxpool and get the logits
        logits = self.softmax(pooled_H)
        if return_embeds:
            return logits, w_embeds
        else:
            return logits
