#!/usr/bin/env python3
"""Helper functions for training classification models"""

from math import ceil
import time
import argparse

import numpy as np
import dynet as dy

import dynn
from dynn.util import num_params

import models
import util


def get_training_args():
    parser = argparse.ArgumentParser()
    # Dynet
    dynn.command_line.add_dynet_args(parser)
    # Data
    data_group = parser.add_argument_group("Data arguments")
    data_group.add_argument("--dataset", default="sst",
                            choices=["sst", "amazon"])
    data_group.add_argument("--data-dir", default="data")
    data_group.add_argument("--reprocess-data", action="store_true")
    data_group.add_argument("--lowercase", action="store_true")
    # Optimization
    optim_group = parser.add_argument_group("Optimization arguments")
    optim_group.add_argument("--batch-size", default=150, type=int)
    optim_group.add_argument("--max-tokens-per-batch", default=4000, type=int)
    optim_group.add_argument("--n-epochs", default=10, type=int)
    optim_group.add_argument("--patience", default=2, type=int)
    optim_group.add_argument("--lr", default=0.001, type=float)
    optim_group.add_argument("--lr-decay", default=0.1, type=float)
    # Model
    model_group = parser.add_argument_group("Model arguments")
    model_group.add_argument("--model-type", default="sopa",
                             choices=models.supported_model_types)
    model_group.add_argument("--model-file", default="sopa_sst.npz")
    model_group.add_argument("--pretrained-embeds", default=None, type=str)
    model_group.add_argument("--freeze-embeds", action="store_true")
    model_group.add_argument("--normalize-embeds", action="store_true")
    # Misc
    misc_group = parser.add_argument_group("Miscellaneous arguments")
    model_group.add_argument("--verbose", action="store_true")
    misc_group.add_argument("--log-file", default=None, type=str)
    misc_group.add_argument("--n-explain", default=10, type=int)
    misc_group.add_argument("--n-top-contrib", default=10, type=int)
    # Parse args to get model type
    args, _ = parser.parse_known_args()
    # Add model specific arguments
    models.add_model_args(args.model_type, parser)
    # Parse again to get all arguments
    args = parser.parse_args()
    return args


def instantiate_network(args, dic, log=None):
    """Create the neural network from command line arguments"""
    log = log or util.Logger(verbose=args.verbose, flush=True)
    # Instantiate the network
    network = models.model_from_args(args.model_type, dic, 2, args)
    # Print some infor about the number of parameters
    log(f"{network.__class__.__name__} model with:")
    log(f"Total parameters: {num_params(network.pc)}")
    log(f" - word embeddings: {num_params(network.pc, params=False)}")
    log(f" - other: {num_params(network.pc, lookup_params=False)}")
    # Load pretrained word embeddings maybe
    if args.pretrained_embeds is not None:
        network.load_pretrained_embeddings(args.pretrained_embeds)
        network.freeze_embeds = args.freeze_embeds
    # normalize to unit norm
    if args.normalize_embeds:
        network.normalize_embeddings()
    return network


def train(args, network, train_batches, dev_batches, log=None):
    """Estimate model parameters on `train_batches`
    with early stopping on`dev_batches`"""
    # Logger
    log = log or util.Logger(verbose=args.verbose, flush=True)
    # Optimizer
    trainer = dy.AdamTrainer(network.pc, alpha=args.lr)
    # Start training
    log("Starting training")
    best_accuracy = 0
    deadline = 0
    running_nll = n_processed = 0
    report_every = ceil(len(train_batches) / 10)
    # Start training
    for epoch in range(1, args.n_epochs+1):
        # Time the epoch
        start_time = time.time()
        for batch, y in train_batches:
            # Renew the computation graph
            dy.renew_cg()
            # Initialize layers
            network.init(test=False, update=True)
            # Compute logits
            logits = network(batch)
            # Loss function
            nll = dy.mean_batches(dy.pickneglogsoftmax_batch(logits, y))
            # Backward pass
            nll.backward()
            # Update the parameters
            trainer.update()
            # Keep track of the nll
            running_nll += nll.value() * batch.batch_size
            n_processed += batch.batch_size
            # Print the current loss from time to time
            if train_batches.just_passed_multiple(report_every):
                avg_nll = running_nll/n_processed
                log(f"Epoch {epoch}@{train_batches.percentage_done():.0f}%: "
                    f"NLL={avg_nll:.3f}")
                running_nll = n_processed = 0
        # End of epoch logging
        avg_nll = running_nll/n_processed
        log(f"Epoch {epoch}@100%: NLL={avg_nll:.3f}")
        log(f"Took {time.time()-start_time:.1f}s")
        log("=" * 20)
        # Validate
        accuracy = evaluate(args, network, dev_batches)
        # Print final result
        log(f"Dev accuracy: {accuracy*100:.2f}%")
        # Early stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            dynn.io.save(network.pc, args.model_file)
            deadline = 0
        else:
            if deadline < args.patience:
                dynn.io.populate(network.pc, args.model_file)
                trainer.learning_rate *= args.lr_decay
                deadline += 1
            else:
                log("Early stopping with best accuracy "
                    f"{best_accuracy*100:.2f}%")
                break
    # Load best model
    dynn.io.populate(network.pc, args.model_file)
    return best_accuracy


def evaluate(args, network, test_batches):
    # Test
    accuracy = 0
    for batch, y in test_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Compute logits
        logits = network(batch)
        # Get prediction
        predicted = logits.npvalue().argmax(axis=0)
        # Accuracy
        accuracy += (predicted == y).sum()
    # Return average accuracy
    return accuracy/test_batches.num_samples


def explain(args, network, explain_batches, log=None):
    # Logger
    log = log or util.Logger(verbose=args.verbose, flush=True)
    # Explaine all predictions
    for batch, y in explain_batches:
        # Renew the computation graph
        dy.renew_cg()
        # Initialize layers
        network.init(test=True, update=False)
        # Trace max matches
        matches = network.max_matches(batch)
        # Print
        for b, (scores, start_pos, end_pos) in enumerate(matches):
            # Retrieve sentence
            sentence = network.dic.string(batch.unpadded_sequences[b])
            # Print sentence
            log("-"*80)
            log(" ".join(sentence))
            # Print top contibuting patterns
            class_weights = network.softmax.W_p.as_array()
            contrib = (class_weights[1] - class_weights[0]) * scores

            def print_pattern(idx):
                """Print a single pattern match"""
                polarity = "positive" if contrib[idx] > 0 else "negative"
                match_str = " ".join([
                    word if pos >= start_pos[idx] and pos <= end_pos[idx]
                    else "_" * len(word)
                    for pos, word in enumerate(sentence)
                ])
                log(
                    f"Pattern {idx} ({polarity})\t"
                    f"{contrib[idx]:.2f}\t{match_str}"
                )

            # log("Top patterns")
            top_contrib = np.abs(contrib).argsort()[-args.n_top_contrib:]
            # for pattern_idx in reversed(top_contrib):
            word_contrib = np.zeros(len(sentence))
            for pattern in top_contrib:
                phrase_slice = slice(start_pos[pattern], end_pos[pattern]+1)
                word_contrib[phrase_slice] += contrib[pattern]

            pos_str = " ".join([
                word if word_contrib[i] > 0 else "_" * len(word)
                for i, word in enumerate(sentence)
            ])
            log(f"Positive contribution: {pos_str}")
            neg_str = " ".join([
                word if word_contrib[i] < 0 else "_" * len(word)
                for i, word in enumerate(sentence)
            ])
            log(f"Negative contribution: {neg_str}")
