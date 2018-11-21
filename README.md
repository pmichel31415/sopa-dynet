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
python main.py \
  --dataset sst \
  --model-type=sopa \
  --embed-dim=50 \
  --dropout=0.1 \
  --pattern-desc=20x3+20x4+20x5 \
  --semiring=log_viterbi \
  --n-epsilon=1 \
  --lr=0.05 \
  --patience=3 \
  --n-epochs=20 \
  --verbose
```

This runs in ~=36s per epoch on my CPU (Core i5 @2.80GHz) and gives me 77.65% test accuracy after 6 epochs (best validation accuracy=77.29%).

Here are sample explanations:


```
Payami tries to raise some serious issues about Iran 's electoral process , but the result is a film that 's about as subtle as a party political broadcast .
Positive contribution: ______ _____ __ _____ ____ _______ issues about ____ __ _________ _______ _ ___ ___ ______ __ _ ____ ____ __ _____ __ subtle as a party political broadcast .
Negative contribution: Payami tries to raise some _______ ______ _____ Iran __ _________ process , but the result is a film that 's about as ______ __ _ _____ _________ _________ _
```

```
A smug and convoluted action-comedy that does n't allow an earnest moment to pass without reminding audiences that it 's only a movie .
Positive contribution: _ ____ ___ __________ _____________ ____ ____ ___ _____ __ _______ ______ to pass without _________ _________ ____ __ __ ____ a movie .
Negative contribution: _ smug and __________ action-comedy that does n't allow an earnest moment __ ____ _______ reminding audiences that it 's only _ _____ _
```

```
An intelligent , multi-layered and profoundly humanist -LRB- not to mention gently political -RRB- meditation on the values of knowledge , education , and the affects of cultural and geographical displacement .
Positive contribution: An intelligent _ _____________ ___ __________ ________ _____ ___ __ _______ ______ _________ _____ __________ __ ___ ______ __ _________ _ _________ _ ___ ___ _______ __ ________ ___ ____________ displacement .
Negative contribution: __ ___________ _ multi-layered and profoundly humanist -LRB- not to mention ______ political -RRB- meditation on the values __ knowledge , education , and the _______ __ ________ ___ ____________ ____________ _
```

## Explanations

Explanations from the SoPa model are obtained by retrieving the maximal matching phrase for each pattern. The contribution of each pattern is defined to be `W_p0 - W_p1`, where `W` is the final `n-patterns x 2` classification matrix. Word `w_i`'s contribution is the sum of the contribution of each pattern whose maximally matching phrase contains `w_i`.