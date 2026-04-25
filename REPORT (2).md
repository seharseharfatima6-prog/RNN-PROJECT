# RNN Student Performance Evaluator — Final Report

## Q1. What is an RNN in your own words?

A Recurrent Neural Network (RNN) is a neural network that has a "memory."
Unlike a standard feedforward network which processes each input independently,
an RNN passes a hidden state vector from one timestep to the next. At each step t,
it combines the current input x_t with the previous hidden state h_{t-1} to
produce a new h_t. This lets the network accumulate context across a sequence —
for example, noticing that a student's quiz scores have been declining over weeks.

## Q2. The three cell types implemented

| Cell       | Gates | Params (input=5, hidden=64) | Strength |
|------------|-------|-----------------------------|---------|
| SimpleRNN  | None  | ~8,800                      | Fast, but vanishing gradients |
| GRU        | 2 (update, reset) | ~25,000           | Good balance of memory and speed |
| LSTM       | 4 (i, f, g, o)    | ~33,000           | Best long-range memory |

All are implemented from scratch in `rnn_core.py` using NumPy only.

## Q3. What function did the RNN learn?

    f(x₁, x₂, x₃, x₄, x₅) → {0=Fail, 1=Pass}

Where each xₜ is the 5-feature vector for semester week t.
The model learned a set of weight matrices (W_xh, W_hh for SimpleRNN;
W_xi/f/g/o, W_hi/f/g/o for LSTM) that together approximate the mapping
from a student's 5-week learning trajectory to their final Pass/Fail outcome.

## Q4. How sequences are constructed

Since the raw data has one row per student (final marks only), we simulate a
5-week semester trajectory using:

    x_t = x_final × (0.55 + 0.45 × t/(T-1))  + ε_t

This means week 1 shows ~55% of final marks, week 5 shows the actual marks,
with Gaussian noise added to simulate realistic weekly variation.
This transforms the (600, 5) tabular dataset into a (600, 5, 5) tensor.

## Q5. Why BPTT (Backpropagation Through Time)?

Standard backprop computes ∂Loss/∂weights for a single forward pass.
In an RNN, the same weights are used at every timestep, so gradients must
be summed across all T timesteps by "unrolling" the network:

    ∂Loss/∂W = Σₜ ∂Lₜ/∂W

This is BPTT. The gradient of the loss w.r.t. W at timestep 1 must pass
through T-1 matrix multiplications, causing gradients to either vanish
(approaching 0) or explode. LSTM/GRU gates solve the vanishing problem.
Gradient clipping (±5.0) solves the explosion problem.

## Q6. Why is scaling critical for RNNs?

In a feedforward network, un-scaled inputs mainly slow convergence.
In an RNN, the same weights are applied at every timestep, so un-scaled
inputs cause the hidden state to grow unboundedly, pushing tanh/sigmoid
into saturation and killing gradients — far more damaging than in MLPs.
StandardScaler (μ=0, σ=1) keeps all activations in the responsive range.

## Q7. Limitations

| Limitation | Detail |
|-----------|--------|
| Synthetic sequences | Weekly trajectories are simulated, not real measured data |
| Short sequences (T=5) | Real RNN benefit requires longer sequences (50+ steps) |
| Small dataset | 600 rows limits generalisation |
| No attention mechanism | Transformer-style attention would weight important weeks more |
| Binary only | Could be extended to grade prediction (regression) |
