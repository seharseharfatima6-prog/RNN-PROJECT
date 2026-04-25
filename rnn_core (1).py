"""
rnn_core.py
===========
Pure-NumPy implementation of:
  • SimpleRNN  — vanilla recurrent cell
  • LSTMCell   — Long Short-Term Memory cell (full gating)
  • GRUCell    — Gated Recurrent Unit cell
  • RNNModel   — wraps any cell into a sequence classifier with
                 Adam optimiser, dropout, L2 regularisation,
                 early stopping, and learning-rate decay.

No TensorFlow / PyTorch needed.
"""

import numpy as np
import pickle


# ─────────────────────────────────────────────────────────────────────────────
# Activation helpers
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(s):          # s is already sigmoid(x)
    return s * (1.0 - s)

def tanh(x):
    return np.tanh(x)

def tanh_grad(t):             # t is already tanh(x)
    return 1.0 - t ** 2

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# SimpleRNN cell
# ─────────────────────────────────────────────────────────────────────────────
class SimpleRNNCell:
    """h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)"""

    def __init__(self, input_size, hidden_size):
        self.H = hidden_size
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.W_xh = np.random.randn(input_size,  hidden_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h  = np.zeros(hidden_size)

    def param_names(self):
        return ['W_xh', 'W_hh', 'b_h']

    def forward(self, x, h_prev):
        raw = x @ self.W_xh + h_prev @ self.W_hh + self.b_h
        h   = tanh(raw)
        cache = (x, h_prev, raw, h)
        return h, h, cache          # (new_h, new_c≡h, cache)

    def backward(self, dh, dc, cache):
        x, h_prev, raw, h = cache
        d_raw = dh * tanh_grad(h)
        dW_xh = x.T    @ d_raw
        dW_hh = h_prev.T @ d_raw
        db_h  = d_raw.sum(axis=0)
        dx    = d_raw  @ self.W_xh.T
        dh_prev = d_raw @ self.W_hh.T
        grads = {'W_xh': dW_xh, 'W_hh': dW_hh, 'b_h': db_h}
        return dx, dh_prev, None, grads


# ─────────────────────────────────────────────────────────────────────────────
# LSTM cell
# ─────────────────────────────────────────────────────────────────────────────
class LSTMCell:
    """
    i = σ(W_xi·x + W_hi·h + b_i)   input gate
    f = σ(W_xf·x + W_hf·h + b_f)   forget gate
    g = tanh(W_xg·x + W_hg·h + b_g) cell gate
    o = σ(W_xo·x + W_ho·h + b_o)   output gate
    c_t = f*c_{t-1} + i*g
    h_t = o * tanh(c_t)
    """
    def __init__(self, input_size, hidden_size):
        self.H = hidden_size
        s = np.sqrt(2.0 / (input_size + hidden_size))
        def W(r,c): return np.random.randn(r,c)*s
        self.W_xi = W(input_size, hidden_size);  self.W_hi = W(hidden_size, hidden_size); self.b_i = np.zeros(hidden_size)
        self.W_xf = W(input_size, hidden_size);  self.W_hf = W(hidden_size, hidden_size); self.b_f = np.ones(hidden_size)   # bias=1 → remember by default
        self.W_xg = W(input_size, hidden_size);  self.W_hg = W(hidden_size, hidden_size); self.b_g = np.zeros(hidden_size)
        self.W_xo = W(input_size, hidden_size);  self.W_ho = W(hidden_size, hidden_size); self.b_o = np.zeros(hidden_size)

    def param_names(self):
        return ['W_xi','W_hi','b_i','W_xf','W_hf','b_f',
                'W_xg','W_hg','b_g','W_xo','W_ho','b_o']

    def forward(self, x, h_prev, c_prev):
        i = sigmoid(x @ self.W_xi + h_prev @ self.W_hi + self.b_i)
        f = sigmoid(x @ self.W_xf + h_prev @ self.W_hf + self.b_f)
        g = tanh   (x @ self.W_xg + h_prev @ self.W_hg + self.b_g)
        o = sigmoid(x @ self.W_xo + h_prev @ self.W_ho + self.b_o)
        c = f * c_prev + i * g
        h = o * tanh(c)
        cache = (x, h_prev, c_prev, i, f, g, o, c)
        return h, c, cache

    def backward(self, dh, dc_next, cache):
        x, h_prev, c_prev, i, f, g, o, c = cache
        tanh_c    = tanh(c)
        dc        = dh * o * tanh_grad(tanh_c) + dc_next
        di        = dc * g  * sigmoid_grad(i)
        df        = dc * c_prev * sigmoid_grad(f)
        dg        = dc * i  * tanh_grad(g)
        do        = dh * tanh_c * sigmoid_grad(o)
        dc_prev   = dc * f

        def grad_gate(dgate, W_x, W_h):
            return (dgate @ W_x.T,
                    dgate @ W_h.T,
                    x.T @ dgate,
                    h_prev.T @ dgate,
                    dgate.sum(axis=0))

        dx  = np.zeros_like(x)
        dh_prev = np.zeros_like(h_prev)
        grads = {}
        for dgate, wx, wh, bname, wxn, whn, bn in [
            (di, self.W_xi, self.W_hi, 'b_i', 'W_xi', 'W_hi', 'b_i'),
            (df, self.W_xf, self.W_hf, 'b_f', 'W_xf', 'W_hf', 'b_f'),
            (dg, self.W_xg, self.W_hg, 'b_g', 'W_xg', 'W_hg', 'b_g'),
            (do, self.W_xo, self.W_ho, 'b_o', 'W_xo', 'W_ho', 'b_o'),
        ]:
            dx      += dgate @ wx.T
            dh_prev += dgate @ wh.T
            grads[wxn] = x.T      @ dgate
            grads[whn] = h_prev.T @ dgate
            grads[bn]  = dgate.sum(axis=0)

        return dx, dh_prev, dc_prev, grads


# ─────────────────────────────────────────────────────────────────────────────
# GRU cell
# ─────────────────────────────────────────────────────────────────────────────
class GRUCell:
    """
    z = σ(W_xz·x + W_hz·h + b_z)   update gate
    r = σ(W_xr·x + W_hr·h + b_r)   reset gate
    n = tanh(W_xn·x + r*(W_hn·h) + b_n) candidate
    h_t = (1-z)*n + z*h_{t-1}
    """
    def __init__(self, input_size, hidden_size):
        self.H = hidden_size
        s = np.sqrt(2.0 / (input_size + hidden_size))
        def W(r,c): return np.random.randn(r,c)*s
        self.W_xz = W(input_size,hidden_size); self.W_hz = W(hidden_size,hidden_size); self.b_z = np.zeros(hidden_size)
        self.W_xr = W(input_size,hidden_size); self.W_hr = W(hidden_size,hidden_size); self.b_r = np.zeros(hidden_size)
        self.W_xn = W(input_size,hidden_size); self.W_hn = W(hidden_size,hidden_size); self.b_n = np.zeros(hidden_size)

    def param_names(self):
        return ['W_xz','W_hz','b_z','W_xr','W_hr','b_r','W_xn','W_hn','b_n']

    def forward(self, x, h_prev, c_prev=None):
        z = sigmoid(x @ self.W_xz + h_prev @ self.W_hz + self.b_z)
        r = sigmoid(x @ self.W_xr + h_prev @ self.W_hr + self.b_r)
        n = tanh   (x @ self.W_xn + r * (h_prev @ self.W_hn) + self.b_n)
        h = (1 - z) * n + z * h_prev
        cache = (x, h_prev, z, r, n)
        return h, h, cache

    def backward(self, dh, dc, cache):
        x, h_prev, z, r, n = cache
        dh_prev = dh * z
        dz  = dh * (h_prev - n) * sigmoid_grad(z)
        dn  = dh * (1 - z) * tanh_grad(n)
        dr  = dn * (h_prev @ self.W_hn) * sigmoid_grad(r)
        dWxn = x.T @ dn;    dWhn = (r * h_prev).T @ dn
        dh_prev += dn @ self.W_hn.T * r + dz @ self.W_hz.T + dr @ self.W_hr.T
        grads = {
            'W_xz': x.T @ dz,     'W_hz': h_prev.T @ dz, 'b_z': dz.sum(0),
            'W_xr': x.T @ dr,     'W_hr': h_prev.T @ dr, 'b_r': dr.sum(0),
            'W_xn': dWxn,         'W_hn': dWhn,          'b_n': dn.sum(0),
        }
        dx = dn @ self.W_xn.T + dz @ self.W_xz.T + dr @ self.W_xr.T
        return dx, dh_prev, None, grads


# ─────────────────────────────────────────────────────────────────────────────
# Adam optimiser state
# ─────────────────────────────────────────────────────────────────────────────
class AdamState:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, l2=1e-4):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.l2    = l2
        self.m     = {}
        self.v     = {}
        self.t     = 0

    def step(self, param_name, param, grad):
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        self.t += 1
        g  = grad + self.l2 * param
        self.m[param_name] = self.beta1 * self.m[param_name] + (1-self.beta1) * g
        self.v[param_name] = self.beta2 * self.v[param_name] + (1-self.beta2) * g**2
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# Full RNN Model
# ─────────────────────────────────────────────────────────────────────────────
class RNNModel:
    """
    Sequence → binary classification model.

    Architecture
    ────────────
    Input  (batch, seq_len, input_size)
        ↓  RNN/LSTM/GRU cell (unrolled over seq_len steps)
    Last hidden state  (batch, hidden_size)
        ↓  Dropout
        ↓  Dense layer  (hidden_size → num_classes)
        ↓  Softmax
    Output (batch, num_classes)
    """

    def __init__(self, cell_type='LSTM', input_size=5, hidden_size=64,
                 num_classes=2, lr=0.001, dropout=0.2, l2=1e-4):
        self.cell_type   = cell_type.upper()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout_rate = dropout

        cell_map = {'SIMPLERNN': SimpleRNNCell, 'LSTM': LSTMCell, 'GRU': GRUCell}
        if self.cell_type not in cell_map:
            raise ValueError(f"cell_type must be one of {list(cell_map)}")
        self.cell = cell_map[self.cell_type](input_size, hidden_size)

        # Output dense layer
        scale = np.sqrt(2.0 / (hidden_size + num_classes))
        self.W_out = np.random.randn(hidden_size, num_classes) * scale
        self.b_out = np.zeros(num_classes)

        self.adam = AdamState(lr=lr, l2=l2)
        self.history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, X, training=True):
        """X: (batch, seq_len, features)"""
        B, T, _ = X.shape
        h = np.zeros((B, self.hidden_size))
        c = np.zeros((B, self.hidden_size))
        self._caches = []

        for t in range(T):
            x_t = X[:, t, :]
            if self.cell_type == 'LSTM':
                h, c, cache = self.cell.forward(x_t, h, c)
            else:
                h, c, cache = self.cell.forward(x_t, h)
            self._caches.append(cache)

        # Dropout on last hidden state
        if training and self.dropout_rate > 0:
            self._drop_mask = (np.random.rand(*h.shape) > self.dropout_rate).astype(float)
            h_drop = h * self._drop_mask / (1 - self.dropout_rate)
        else:
            self._drop_mask = np.ones_like(h)
            h_drop = h

        self._h_last  = h
        self._h_drop  = h_drop

        logits = h_drop @ self.W_out + self.b_out
        probs  = softmax(logits)
        return probs

    # ── loss ─────────────────────────────────────────────────────────────────
    def cross_entropy(self, probs, labels):
        B = len(labels)
        loss = -np.log(probs[np.arange(B), labels] + 1e-12).mean()
        return loss

    # ── backward ─────────────────────────────────────────────────────────────
    def backward(self, probs, labels):
        B = len(labels)
        # Gradient of softmax + cross-entropy
        d_logits = probs.copy()
        d_logits[np.arange(B), labels] -= 1
        d_logits /= B

        # Dense layer gradients
        dW_out = self._h_drop.T @ d_logits
        db_out = d_logits.sum(axis=0)
        dh     = (d_logits @ self.W_out.T) * self._drop_mask

        # BPTT through all timesteps
        dc = np.zeros_like(dh)
        all_cell_grads = {}

        for t in reversed(range(len(self._caches))):
            cache = self._caches[t]
            if self.cell_type == 'LSTM':
                _, dh, dc, cell_grads = self.cell.backward(dh, dc, cache)
            else:
                _, dh, dc, cell_grads = self.cell.backward(dh, dc, cache)
            for k, v in cell_grads.items():
                if k not in all_cell_grads:
                    all_cell_grads[k] = np.zeros_like(v)
                all_cell_grads[k] += v

        # Clip gradients
        clip = 5.0
        for k in all_cell_grads:
            np.clip(all_cell_grads[k], -clip, clip, out=all_cell_grads[k])

        # Apply Adam updates — cell params
        for pname in self.cell.param_names():
            if pname in all_cell_grads:
                p   = getattr(self.cell, pname)
                new = self.adam.step(f'cell_{pname}', p, all_cell_grads[pname])
                setattr(self.cell, pname, new)

        # Apply Adam updates — output layer
        self.W_out = self.adam.step('W_out', self.W_out, dW_out)
        self.b_out = self.adam.step('b_out', self.b_out, db_out)

    # ── predict ──────────────────────────────────────────────────────────────
    def predict_proba(self, X):
        return self.forward(X, training=False)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    # ── train ─────────────────────────────────────────────────────────────────
    def fit(self, X_train, y_train, X_val, y_val,
            epochs=150, batch_size=32,
            early_stopping_patience=20,
            lr_decay_factor=0.5, lr_decay_patience=10,
            verbose=True):

        best_val_loss = np.inf
        best_weights  = None
        no_improve    = 0
        lr_no_improve = 0

        for epoch in range(1, epochs+1):
            # Shuffle
            idx = np.random.permutation(len(X_train))
            X_s, y_s = X_train[idx], y_train[idx]

            # Mini-batch training
            train_loss = 0.0
            for start in range(0, len(X_s), batch_size):
                Xb = X_s[start:start+batch_size]
                yb = y_s[start:start+batch_size]
                probs = self.forward(Xb, training=True)
                loss  = self.cross_entropy(probs, yb)
                train_loss += loss * len(Xb)
                self.backward(probs, yb)

            train_loss /= len(X_train)
            train_acc   = (self.predict(X_train) == y_train).mean()

            # Validation
            val_probs = self.predict_proba(X_val)
            val_loss  = self.cross_entropy(val_probs, y_val)
            val_acc   = (self.predict(X_val) == y_val).mean()

            self.history['train_loss'].append(float(train_loss))
            self.history['val_loss'].append(float(val_loss))
            self.history['train_acc'].append(float(train_acc))
            self.history['val_acc'].append(float(val_acc))

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:>4d} | "
                      f"Train loss: {train_loss:.4f}  acc: {train_acc*100:.1f}% | "
                      f"Val loss: {val_loss:.4f}  acc: {val_acc*100:.1f}%")

            # Early stopping
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_weights  = self._snapshot()
                no_improve    = 0
                lr_no_improve = 0
            else:
                no_improve    += 1
                lr_no_improve += 1

            if lr_no_improve >= lr_decay_patience:
                self.adam.lr *= lr_decay_factor
                lr_no_improve = 0
                if verbose:
                    print(f"  → LR decayed to {self.adam.lr:.6f}")

            if no_improve >= early_stopping_patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

        # Restore best
        if best_weights:
            self._restore(best_weights)
        if verbose:
            print(f"\n✅ Best val loss: {best_val_loss:.4f}")

    # ── weight snapshot / restore ─────────────────────────────────────────────
    def _snapshot(self):
        snap = {'W_out': self.W_out.copy(), 'b_out': self.b_out.copy()}
        for pname in self.cell.param_names():
            snap[f'cell_{pname}'] = getattr(self.cell, pname).copy()
        return snap

    def _restore(self, snap):
        self.W_out = snap['W_out'].copy()
        self.b_out = snap['b_out'].copy()
        for pname in self.cell.param_names():
            setattr(self.cell, pname, snap[f'cell_{pname}'].copy())

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
