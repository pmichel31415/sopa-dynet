# Soft Patterns in DyNet

This is a Dynet reimplementation of [SoPa: Bridging CNNs, RNNs, and Weighted Finite-State Machines](https://arxiv.org/abs/1805.06061) by Schwartz et al. (2018). The original code from the authors can be found [here](https://github.com/Noahs-ARK/soft_patterns).

## Requirements

See the `requirements.txt` file but basically you will need

- Python>=3.6
- [DyNet](https://github.com/clab/dynet) for efficient computation and autograd on CPU and GPU
- [DyNN](https://github.com/pmichel31415/dynn) wrappers I wrote around DyNet.

I recommend setting up a virtual environment:

```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

## Getting Started

You can train a small SoPa model on CPU by calling

```bash
python sst.py \
  --model-type=sopa \
  --embed-dim=50 \
  --dropout=0.0 \
  --pattern-desc=20x3+20x4+20x5 \
  --semiring=log_viterbi \
  --n-epsilon=1 \
  --lr=0.001 \
  --patience=3 \
  --n-epochs=20 \
  --verbose
```

This runs in ~=16s per epoch on my CPU (Core i5 @2.80GHz) and gives me 75.89% test accuracy after 6 epochs (best validation accuracy=75.80%).

Here are sample explanations:

```
... silly humbuggery ...
Positive contribution: ... _____ __________ ___
Negative contribution: ___ silly humbuggery ...
```

```
But they do n't fit well together and neither is well told .
Positive contribution: ___________ , __ ____________ _ movie .
Negative contribution: Forgettable _ if good-hearted , _____ _
```

```
A rollicking ride , with jaw-dropping action sequences , striking villains , a gorgeous color palette , astounding technology , stirring music and a boffo last hour that leads up to a strangely sinister happy ending .
Positive contribution: A rollicking ____ _ ____ ____________ ______ _________ _ ________ ________ _ _ ________ _____ _______ _ __________ __________ _ ________ _____ ___ _ _____ ____ ____ ____ _____ __ __ _ _________ ________ happy ending .
Negative contribution: _ __________ ride , ____ ____________ ______ _________ , striking ________ _ _ ________ _____ _______ _ __________ __________ _ stirring music ___ _ _____ last hour that leads up to a strangely sinister _____ ______ _
```

## Explanations

Explanations from the SoPa model are obtained by retrieving the maximal matching phrase for each pattern. The contribution of each pattern is defined to be `W_p0 - W_p1`, where `W` is the final `n-patterns x 2` classification matrix. Word `w_i`'s contribution is the sum of the contribution of each pattern whose maximally matching phrase contains `w_i`.