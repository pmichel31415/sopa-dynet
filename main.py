#!/usr/bin/env python3
"""Script for training models on SST"""

import dynn

from util import Logger
from training import (
    train,
    evaluate,
    explain,
    get_training_args,
    instantiate_network
)
from datasets import get_dataset

# For reproducibility
dynn.set_random_seed(31415)


def main():
    args = get_training_args()
    # Logger
    log = Logger(file=args.log_file, verbose=args.verbose, flush=True)
    # Prepare data
    batches, dic = get_dataset(args, log)
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
