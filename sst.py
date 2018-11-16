#!/usr/bin/env python3
"""Script for training models on SST"""

import numpy as np

import dynn
from dynn.data import sst
from dynn.data import preprocess
from dynn.data import Dictionary
from dynn.data.batching import PaddedSequenceBatches

from util import Logger
from training import (
    train,
    evaluate,
    explain,
    get_training_args,
    instantiate_network
)

# For reproducibility
dynn.set_random_seed(31415)


def load_and_prepare_sst(args, log=None):
    log = log or Logger(verbose=args.verbose, flush=True)
    # Download SST
    sst.download_sst(args.data_dir)
    # Load the data
    log("Loading the SST data")
    data = sst.load_sst(args.data_dir, terminals_only=True, binary=True)
    train_x, train_y = data["train"]
    log(f"{len(train_x)} training samples")
    dev_x, dev_y = data["dev"]
    log(f"{len(dev_x)} dev samples")
    test_x, test_y = data["test"]
    log(f"{len(test_x)} test samples")
    # Lowercase
    if args.lowercase:
        log("Lowercasing")
        train_x, dev_x, test_x = preprocess.lowercase([train_x, dev_x, test_x])
    # Learn the dictionary
    log("Building the dictionary")
    dic = Dictionary.from_data(train_x)
    dic.freeze()
    dic.save("sst.dic")
    log(f"{len(dic)} symbols in the dictionary")
    # Numberize the data
    log("Numberizing")
    train_x = dic.numberize(train_x)
    dev_x = dic.numberize(dev_x)
    test_x = dic.numberize(test_x)
    # Create the batch iterators
    log("Creating batch iterators")
    batches = {}
    batches["train"] = PaddedSequenceBatches(
        train_x, train_y, dic, max_samples=args.batch_size
    )
    batches["dev"] = PaddedSequenceBatches(
        dev_x, dev_y, dic, max_samples=32, shuffle=False
    )
    batches["test"] = PaddedSequenceBatches(
        test_x, test_y, dic, max_samples=32, shuffle=False
    )
    # Select a subset of the training data for showcasing explanations
    explain_subset = np.random.choice(
        len(train_y),
        size=args.n_explain,
        replace=False
    )
    batches["explain"] = PaddedSequenceBatches(
        [train_x[i] for i in explain_subset],
        [train_y[i] for i in explain_subset],
        dic,
        max_samples=5,
    )
    return batches, dic


def main():
    args = get_training_args()
    # Logger
    log = Logger(file=args.log_file, verbose=args.verbose, flush=True)
    # Prepare data
    batches, dic = load_and_prepare_sst(args, log)
    # Create model
    network = instantiate_network(args, dic, log)
    # Train model
    train(args, network, batches["train"], batches["dev"], log)
    # Test
    test_accuracy = evaluate(args, network, batches["test"])
    # Print final result
    log(f"Test accuracy: {test_accuracy*100:.2f}%")
    # Explain if the model is SoPa
    if args.model_type == "sopa":
        explain(args, network, batches["explain"], log)


if __name__ == "__main__":
    main()
