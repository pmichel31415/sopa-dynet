#!/usr/bin/env python3
import numpy as np


def load_pretrained_embeddings(
    word_embeddings,
    dic,
    filename,
    lowercase=False,
):
    """Load word embeddings from a glove-style file"""
    # Load vectors word by word
    with open(filename) as f:
        for line in f:
            word, vector_string = line.strip().split(" ", 1)
            if lowercase:
                word = word.lower()
            if word in dic.indices:
                wid = dic.index(word)
                vector = np.fromstring(vector_string, sep=" ")
                word_embeddings.init_row(wid, vector)


def normalize_embeddings(embeddings, norm=1.0):
    embed_matrix = embeddings.as_array()
    norms = np.linalg.norm(embed_matrix, axis=-1).reshape(-1, 1)
    norms = np.where(norms <= 0, 1e-20, norms)
    embeddings.init_from_array(embed_matrix / norms * norm)
