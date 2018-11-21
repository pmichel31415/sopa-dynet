#!/usr/bin/env python3
import numpy as np

from dynn.data import sst, amazon
from dynn.data import caching
from dynn.data import preprocess
from dynn.data import Dictionary
from dynn.data.batching import PaddedSequenceBatches

from util import Logger


def get_dataset(args, log=None):
    if args.dataset == "sst":
        dataset = load_and_prepare_sst(args, log)
    elif args.dataset == "amazon":
        dataset = load_and_prepare_amazon(args, log)
    return dataset


@caching.cached_to_file("data/sst.cached.bin")
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
        train_x, train_y, dic, max_samples=args.batch_size, max_tokens=args.max_tokens_per_batch
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


@caching.cached_to_file("data/amazon.cached.bin")
def load_and_prepare_amazon(args, log=None):
    log = log or Logger(verbose=args.verbose, flush=True)
    # Download amazon
    amazon.download_amazon(args.data_dir)
    # Load the data
    log("Loading the amazon data")
    data = amazon.load_amazon(args.data_dir, tok=True, size="25k")
    # Split in train/dev (TODO: check correctness)
    data["dev"] = [x[-5000:] for x in data["train"]]
    data["train"] = [x[:-5000] for x in data["train"]]
    # Print data size
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
    dic.save("amazon.dic")
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
        train_x, train_y, dic, max_samples=args.batch_size, max_tokens=args.max_tokens_per_batch
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
