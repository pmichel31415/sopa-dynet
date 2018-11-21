#!/usr/bin/env python3
import dynet as dy

from dynn import activations


class Semiring(object):
    """All operations defining a semiring:

    zero (function): Return an expression full of zeros given a dimension
    one (function): Return an expression full of ones given a dimension
    plus (function): Addition in the semiring
    times (function): Multiplication in the semiring
    encode (function): Encode from reals to the semiring
        (eg when you want to convert to log space)
    decode (function): Decode to the reals
    """

    def __init__(self, zero, one, plus, times, encode, decode):
        self.zero = zero
        self.one = one
        self.plus = plus
        self.times = times
        self.encode = encode
        self.decode = decode


def neg_inf(dim, batch_size=1):
    """Expression for negative infinity (actually -1000)"""
    return dy.zeros(dim, batch_size=batch_size) - 1000


def logadd(x, y):
    """Binary logsumexp"""
    return dy.logsumexp([x, y])


semirings = {
    # Probability semiring
    "prob": Semiring(
        zero=dy.zeros,
        one=dy.ones,
        plus=dy.Expression.__add__,
        times=dy.cmult,
        encode=activations.sigmoid,
        decode=activations.identity,
    ),
    # Log semiring
    "log": Semiring(
        zero=neg_inf,
        one=dy.zeros,
        plus=logadd,
        times=dy.Expression.__add__,
        encode=dy.log_sigmoid,
        decode=dy.exp,
    ),
    # Tropical semiring (max-plus)
    "tropical": Semiring(
        zero=neg_inf,
        one=dy.zeros,
        plus=dy.bmax,
        times=dy.Expression.__add__,
        encode=activations.identity,
        decode=activations.identity,
    ),
    # Viterbi semiring (max-product)
    "viterbi": Semiring(
        zero=neg_inf,
        one=dy.ones,
        plus=dy.bmax,
        times=dy.cmult,
        encode=activations.sigmoid,
        decode=activations.identity,
    ),
    # Viterbi semiring (in log space)
    "log_viterbi": Semiring(
        zero=neg_inf,
        one=dy.zeros,
        plus=dy.bmax,
        times=dy.Expression.__add__,
        encode=dy.log_sigmoid,
        decode=dy.exp,
    ),
}
