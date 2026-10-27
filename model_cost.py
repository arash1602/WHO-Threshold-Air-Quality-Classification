"""Inference cost of every pooled configuration, computed analytically (no PyTorch).

    python model_cost.py      # -> tables/model_cost.txt, tables/model_cost.json

For one 24-hour window (24 time steps, 5 channels) and a single-logit output:

params   trainable parameters, counted exactly as PyTorch counts them (GRU input and
         hidden biases separately; weight-normalized TCN convolutions carry an extra
         gain vector ``weight_g`` per convolution). Checked against the counts the
         sweep reports, via make_tables.n_params when PyTorch is present, and
         against the paper's tables otherwise.
MACs     multiply-accumulates of the matrix and convolution products for a full
         forward pass over the window, which dominate the cost; activations, gates'
         element-wise products and biases are excluded. The TCN and the FUSION
         convolution are counted over all 24 positions, as the reference
         implementation computes them; a streaming implementation could compute
         only the last position of the final block.
weights  bytes to store the parameters at 32-bit float and 8-bit integer precision.
act      peak bytes of 32-bit activations that must be live at once, as a
         lower bound for working RAM: for recurrent layers one hidden state per
         layer (the window is processed step by step); for convolutions the whole
         24-step feature map of the widest layer pair.

The logistic regression is listed for reference: 6 parameters and 5 MACs on window
means (plus the 120 additions that form the means).
"""
from __future__ import annotations

import json
import os

T = 24          # context length (hours)
C_IN = 5        # input channels
ARCHS = ("GRU", "TCN", "FUSION")
HORIZONS = (1, 12, 24)


def gru_cost(c_in, hidden, layers):
    params = macs = 0
    for i in range(layers):
        x = c_in if i == 0 else hidden
        params += 3 * (hidden * x + hidden * hidden + 2 * hidden)
        macs += T * 3 * (hidden * x + hidden * hidden)
    return params, macs


def gru_model(hp):
    p, m = gru_cost(C_IN, hp["hidden_size"], hp["num_layers"])
    h = hp["hidden_size"]
    act = 4 * h * hp["num_layers"] * 2          # current and next state per layer
    return p + h + 1, m + h, act


def tcn_model(hp):
    chans, k = list(hp["tcn_channels"]), hp["kernel_size"]
    params = macs = 0
    act = 0
    prev = C_IN
    for c in chans:
        # conv1 prev->c, conv2 c->c, each with bias and weight-norm gain g (c values)
        params += (c * prev * k + c + c) + (c * c * k + c + c)
        macs += T * (c * prev * k + c * c * k)
        if prev != c:
            params += c * prev + c
            macs += T * c * prev
        act = max(act, 4 * T * (prev + 2 * c))   # block input, hidden map, output
        prev = c
    return params + prev + 1, macs + prev, act


def fusion_model(hp):
    cc, k, gh, fd, nl = (hp["cnn_channels"], hp["kernel_size"], hp["gru_hidden"],
                         hp["fusion_dim"], hp["num_layers"])
    params = cc * C_IN * k + cc
    macs = T * cc * C_IN * k
    params += cc * fd + fd                         # cnn_proj
    macs += cc * fd
    gp, gm = gru_cost(C_IN, gh, nl)
    params += gp
    macs += gm
    params += gh * fd + fd                         # gru_proj
    macs += gh * fd
    params += 2 * fd * fd + fd                     # gate
    macs += 2 * fd * fd
    params += fd + 1                               # head
    macs += fd
    act = 4 * (T * cc + 2 * gh * nl + 3 * fd)
    return params, macs, act


COST = {"GRU": gru_model, "TCN": tcn_model, "FUSION": fusion_model}

# Parameter counts printed in the paper's tables, used as the check when PyTorch
# (and so make_tables.n_params) is unavailable.
PAPER_PARAMS = {
    ("single", 1): {"TCN": 16449, "FUSION": 31937, "GRU": 349185},
    ("single", 12): {"TCN": 63617, "FUSION": 16801, "GRU": 22785},
    ("single", 24): {"TCN": 42817, "GRU": 22785, "FUSION": 20929},
    ("multi", 1): {"GRU": 151041, "TCN": 15393, "FUSION": 21569},
    ("multi", 12): {"GRU": 2753, "TCN": 42817, "FUSION": 20001},
    ("multi", 24): {"GRU": 10113, "TCN": 42817, "FUSION": 19713},
}


def kb(b):
    return b / 1024


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    rd = os.path.join(here, "tuning_results")
    try:
        import make_tables as mt          # exact count through PyTorch when present
        mt.n_params("GRU", {"hidden_size": 8, "num_layers": 1, "dropout": 0.0})
        exact = mt.n_params
    except Exception:
        exact = None
    out, res = [], {}
    say = out.append
    say(f"{'design':6s} {'h':>3s} {'arch':7s} {'params':>8s} {'kMAC':>9s} "
        f"{'fp32 KB':>8s} {'int8 KB':>8s} {'act KB':>7s}  check")
    bad = 0
    for exp in ("single", "multi"):
        for h in HORIZONS:
            for a in ARCHS:
                with open(os.path.join(rd, f"h{h}", f"best_{a}_{exp}_ctx24_blocked_logz.json"),
                          encoding="utf-8") as f:
                    hp = json.load(f)["hyperparameters"]
                p, m, act = COST[a](hp)
                ref = exact(a, hp) if exact else PAPER_PARAMS[(exp, h)][a]
                ok = p == ref
                bad += not ok
                res[f"{exp}-h{h}-{a}"] = {"hp": hp, "params": p, "macs": m,
                                          "fp32_bytes": 4 * p, "int8_bytes": p, "act_bytes": act}
                say(f"{exp:6s} {h:3d} {a:7s} {p:8d} {m / 1e3:9.1f} {kb(4 * p):8.1f} "
                    f"{kb(p):8.1f} {kb(act):7.1f}  {'ok' if ok else f'MISMATCH (expected {ref})'}")
    say(f"\nparameter-count mismatches: {bad}")
    say("logistic regression: 6 params, 5 MACs on window means (+120 additions), 24 B fp32")
    os.makedirs(os.path.join(here, "tables"), exist_ok=True)
    with open(os.path.join(here, "tables", "model_cost.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(here, "tables", "model_cost.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\n".join(out))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
