#!/usr/bin/env python3


def load_pretrained_embeddings(
    word_embeddings,
    dic,
    filename,
    lowercase=False,
    renormalize=True
):
    """Load word embeddings from a glove-style file"""
    # Record the original std/mean of the word embeddings
    if renormalize:
        prev_embeds = word_embeddings.as_array()
        mean = prev_embeds.mean()
        std = prev_embeds.std()
    # Load vectors word by word
    with open(filename) as f:
        for line in f:
            fields = line.strip().split(" ")
            word = fields[0]
            if lowercase:
                word = word.lower()
            if word in dic.indices:
                wid = dic.index(word)
                vector = [float(x) for x in fields[1:]]
                word_embeddings.init_row(wid, vector)
    # Renormalize to the same std/mean as the original word embeddings
    if renormalize:
        new_embeds = word_embeddings.as_array()
        mean_ = new_embeds.mean()
        std_ = new_embeds.std()
        whitened_embeds = (new_embeds - mean_) / std_
        adjusted_embeds = std * whitened_embeds + mean
        word_embeddings.init_from_array(adjusted_embeds)
