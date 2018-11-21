#!/usr/bin/env python3
import re
import numpy as np
import dynet as dy

from dynn.layers import ParametrizedLayer
from dynn.layers import Affine
from dynn.layers import Embeddings
from dynn.layers import Parallel

from dynn.operations import seq_mask

from dynn.data.batching import SequenceBatch

from .semiring import semirings
from .base_classifier import SentenceClassifier

is_pattern_desc = re.compile(
    r"([^0][0-9]*)x([^0][0-9]*)(\+([^0][0-9]*)x([^0][0-9]*))*"
)


def bargmax(x, y):
    """binary argmax"""
    stacked = np.stack([x.npvalue(), y.npvalue()])
    return stacked.argmax(axis=0)


class SoPaLayer(ParametrizedLayer):
    """One "layer" of soft patterns.

    This represents one collection of soft patterns of the same number of
    states and processes them in parallel.
    """

    def __init__(self, pc, sr, dx, dh, ns, n_eps, sl_scale, dropout):
        super(SoPaLayer, self).__init__(pc, "sopa-layer")
        # Hyperparameters
        self.sr = sr  # Semiring
        self.dx = dx  # Word embedding dim
        self.dh = dh  # Number of patterns
        self.ns = ns  # Number of states per pattern
        self.n_eps = n_eps  # Number of epsilon transitions
        self.sl_scale = sl_scale  # Scale for self loops
        self.dropout = dropout  # Dropout rate
        # Transition matrices
        self.next_state = Affine(self.pc, dx, (ns - 1) * dh,
                                 dropout=dropout, activation=self.sr.encode)
        self.same_state = Affine(self.pc, dx, ns * dh,
                                 dropout=dropout, activation=self.sr.encode)
        self.epsilon_p = self.pc.add_parameters((ns - 1, dh), name="eps")
        # Keep track of the addition op
        self.plus_op = self.sr.plus
        self.zero_val = self.sr.zero(1).scalar_value()
        self.one_val = self.sr.one(1).scalar_value()

    def init(self, test=False, update=True):
        self.next_state.init(test=test, update=update)
        self.same_state.init(test=test, update=update)
        self.epsilon = self.sr.encode(self.epsilon_p.expr(update))
        self.ones = self.sr.one(self.ns * self.dh)
        self.zeros = self.sr.zero((1, self.dh))

    def eps(self, h, n=1, start_idx=None, t=None):
        """Execute epsilon transitions"""

        def _one_epsilon(h):
            """One epsilon transition"""
            return dy.concatenate([
                self.zeros,
                self.sr.times(h[:self.ns - 1], self.epsilon)
            ])

        explain = start_idx is not None and t is not None
        h_eps = h
        for _ in range(n):
            h_eps = _one_epsilon(h)
            if explain:
                do_eps = bargmax(h, h_eps)
                start_idx = self.advance_start_pos(start_idx, do_eps, t)
            h = self.sr.plus(h, h_eps)

        if explain:
            return h, start_idx
        else:
            return h

    def get_batch_masks(self, N, bsz, lengths):
        """Get masks for a batch of sentences with different lengths"""
        return seq_mask(
            N,
            lengths or [N for _ in range(bsz)],
            self.one_val,
            self.zero_val,
        )

    def get_transitions(self, x):
        """Compute transition matrices from word vectors"""
        _, N = x.dim()[0]
        # Diagonal of the transition matrix at each step
        # Here this has shape dh*ns x N x bsz
        T_ii = self.same_state(x)
        # Scale
        if self.sl_scale != 1:
            scale = self.sr.encode(dy.scalarInput(self.sl_scale))
            T_ii = self.sr.times(self.T_ii, scale)
        # Split into a list of N dh*ns x bsz expressions
        T_ii = [dy.pick(T_ii, t, 1) for t in range(N)]
        # Reshape each expression into a ns x dh x bsz batched matrix
        T_ii = [dy.reshape(T_ii[t], (self.ns, self.dh)) for t in range(N)]
        # Off-diagonal of the transition matrix at each step
        # Here this has shape dh*ns x N x bsz
        T_ij = self.next_state(x)
        # Split into a list of N dh*ns x bsz expressions
        T_ij = [dy.pick(T_ij, t, 1) for t in range(N)]
        # Reshape each expression into a ns x dh x bsz batched matrix
        T_ij = [dy.reshape(T_ij[t], (self.ns-1, self.dh)) for t in range(N)]
        return T_ii, T_ij

    def initial_scores(self, batch_size=1):
        """Scores of each state before processing the sentence"""
        scores_0 = dy.concatenate([
            # The initial state has score 1
            self.sr.one((1, self.dh), batch_size=batch_size),
            # All others have scores 0
            self.sr.zero((self.ns - 1, self.dh), batch_size=batch_size)
        ])
        return scores_0

    def __call__(self, x, lengths=None, explain=False):
        # Shape
        _, N = x.dim()[0]
        bsz = x.dim()[1]
        # If we want explanation
        if explain:
            # "Viterbize" the semiring (make the addition be max)
            self.viterbize_semiring()
            # Keep track of the start and end index of the top matches
            # start_idx_t[i, h] is the start idx of the top match to pattern
            # h that is currently in state i
            start_pos_t = np.zeros((self.ns, self.dh), dtype=int)
            # start_pos[h] is the start idx of the best match for pattern h
            start_pos = np.zeros(self.dh, dtype=int)
            # end_pos[h] is the end idx of the best match for pattern h
            stop_pos = np.zeros(self.dh, dtype=int)
        # Score for each state at set t
        scores_t = self.initial_scores(batch_size=bsz)
        # Score of the final state
        score_final = scores_t[self.ns - 1]
        # Masks for padded tokens
        masks = self.get_batch_masks(N, bsz, lengths)
        # Compute transition matrices.
        T_ii, T_ij = self.get_transitions(x)
        # iterate over timesteps
        for t in range(N):
            # Execute n_eps epsilon transition
            if explain:
                scores_t, start_pos_t = self.eps(
                    scores_t,
                    n=self.n_eps,
                    start_idx=start_pos_t,
                    t=t,
                )
            else:
                scores_t = self.eps(scores_t, n=self.n_eps)

            # Score of staying in the same state
            score_t_same = self.sr.times(scores_t, T_ii[t])
            # Score of going to the next state
            score_t_next = dy.concatenate([
                self.sr.one((1, self.dh)),
                self.sr.times(scores_t[:self.ns-1], T_ij[t])
            ])
            # Execute self loop/next state transitions
            scores_t = self.sr.plus(score_t_same, score_t_next)
            # Update pointer to start position
            if explain:
                # Do we advance to the next state or not?
                advance = bargmax(score_t_same, score_t_next)
                start_pos_t = self.advance_start_pos(start_pos_t, advance, t)
            # Score of the final state at this step for each pattern
            score_final_t = scores_t[self.ns - 1]
            # Masking if we are batching things with padding
            masked_score_final_t = self.sr.times(masks[t], score_final_t)
            # Update start and end position of the best match
            if explain:
                # Is the best match for each pattern ending here?
                stop_here = bargmax(score_final, masked_score_final_t)
                # If so, update the start and final position
                start_pos = np.where(stop_here, start_pos_t[-1], start_pos)
                stop_pos = np.where(stop_here, t, stop_pos)
            # Update best score
            score_final = self.sr.plus(masked_score_final_t, score_final)
        # Return everything
        if explain:
            # Restor the semiring + operation
            self.de_viterbize_semiring()
            # Returns scores and start/end indices of their best match
            return self.sr.decode(score_final), start_pos, stop_pos
        else:
            return self.sr.decode(score_final)

    def advance_start_pos(self, start_pos, advance, t):
        """Shift the starting position pointer for the best match"""
        # Current position
        this_pos = np.full((1, self.dh), t)
        # New start position assuming we go to the next state
        new_start_pos = np.concatenate([this_pos, start_pos[:-1]])
        return np.where(advance == 1, new_start_pos, start_pos)

    def viterbize_semiring(self):
        """This "viterbizes" the semiring, ie it replaces the addition
        with max"""
        self.sr.plus = dy.bmax

    def de_viterbize_semiring(self):
        """This returns the semiring's addition to the original addition"""
        self.sr.plus = self.plus_op


def parse_pattern_description(pattern_desc):
    # Check that the pattern description is valid
    if not is_pattern_desc.match(pattern_desc):
        raise ValueError(f"Invalid pattern description {pattern_desc}")
    # Parse it
    pattern_shapes = []
    for patterns in pattern_desc.split("+"):
        num, size = patterns.split("x")
        pattern_shapes.append((int(num), int(size)))
    return pattern_shapes


class SoPa(SentenceClassifier):
    """Soft Patterns model.

    Aggregates multiple patterns of various state spaces.
    """

    def __init__(
        self,
        dic,
        sr,
        dx,
        pattern_desc,
        n_eps,
        sl_scale,
        nc,
        dropout
    ):
        # Hyperparameters
        self.dic = dic  # Dictionary
        self.sr = sr  # Semiring
        self.dx = dx  # Word embedding dim
        self.pattern_shapes = parse_pattern_description(pattern_desc)
        self.n_patterns = sum(n for n, _ in self.pattern_shapes)
        self.n_eps = n_eps  # Number of epsilon transitions
        self.sl_scale = sl_scale  # Self-loop scale
        self.nc = nc  # Number of classes
        self.dropout = dropout  # dropout (duh...)
        # Master parameter collection
        self.pc = dy.ParameterCollection()
        # Word embeddings
        self.embed = Embeddings(self.pc, dic, dx, pad_mask=0.0)
        # SoPas
        self.sopa_layers = []
        for dh, ns in self.pattern_shapes:
            sopa_layer = SoPaLayer(self.pc, sr, dx, dh, ns,
                                   n_eps, sl_scale, dropout)
            self.sopa_layers.append(sopa_layer)
        self.sopas = Parallel(*self.sopa_layers)
        # Softmax layer
        self.softmax = Affine(self.pc, self.n_patterns, nc, dropout=dropout)

    @staticmethod
    def add_args(parser):
        sopa_group = parser.add_argument_group("SoPa specific arguments")
        sopa_group.add_argument("--embed-dim", default=100,
                                help="Word embedding dim", type=int)
        sopa_group.add_argument("--pattern-desc", default="50x4",
                                help="Pattern descriptions", type=str)
        sopa_group.add_argument("--n-epsilon", default=1, type=int,
                                help="Number of epsilon transitions")
        sopa_group.add_argument("--self-loop-scale", default=1, type=int,
                                help="Multiply the self loop by this. "
                                ">1 encourages self loop (longer matches), "
                                "whereas <1 discourages self loops")
        sopa_group.add_argument("--semiring", default="maxplus",
                                help="Semiring", type=str,
                                choices=semirings.keys())
        sopa_group.add_argument("--dropout", default=0.1,
                                help="Dropout", type=float)

    @staticmethod
    def from_args(dic, num_classes, args):
        return SoPa(
            dic,
            semirings[args.semiring],
            args.embed_dim,
            args.pattern_desc,
            args.n_epsilon,
            args.self_loop_scale,
            num_classes,
            args.dropout,
        )

    def get_word_embeddings(self):
        """Return the word embedding lookup parameters"""
        return self.embed.params

    def init(self, test=False, update=True):
        frozen_embeds = getattr(self, "freeze_embeds", False)
        self.embed.init(test=test, update=update and not frozen_embeds)
        self.sopas.init(test=test, update=update)
        self.softmax.init(test=test, update=update)

    def __call__(self, batch, return_embeds=False):
        """Return logits for each class"""
        # Handle simple sequence
        if not isinstance(batch, SequenceBatch):
            return self.__call__(SequenceBatch(batch), return_embeds)
        # Embed the f out of the inputs
        w_embeds = dy.transpose(self.embed(batch.sequences))
        # Initial and final state
        h = self.sopas(w_embeds, lengths=batch.lengths)
        # Logits
        logits = self.softmax(h)
        # Return with embeddings
        if return_embeds:
            return logits, w_embeds
        else:
            return logits

    def max_matches(self, batch):
        """Top match for each pattern on this batch"""
        # Handle simple sequence
        if not isinstance(batch, SequenceBatch):
            return self.__call__(SequenceBatch(batch))
        # Batched processing is not handled yet
        if batch.batch_size > 1:
            return [self.max_matches(batch[b])
                    for b in range(batch.batch_size)]
        # Embed the f out of the inputs
        w_embeds = dy.transpose(self.embed(batch.sequences))
        # Max matches for all patterns
        each_layer_matches = [layer(w_embeds, explain=True)
                              for layer in self.sopas.layers]
        scores, start_pos, stop_pos = zip(*each_layer_matches)
        # Concatenate together and return
        scores = np.concatenate([score.npvalue() for score in scores])
        start_pos = np.concatenate(start_pos)
        stop_pos = np.concatenate(stop_pos)
        return scores, start_pos, stop_pos
