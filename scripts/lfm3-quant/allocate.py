"""Greedy bit allocation from measured sensitivities.

Groups are (regex, element count, {type: measured KLD cost}); starting from
every group at its highest-quality option, repeatedly take the downgrade with
the smallest predicted KLD increase per byte saved until the budget is met.
Prints an AMQL to-gguf recipe.

usage: allocate.py <budget-MB> [--metric chat|ood|both] > recipe.txt"""
import csv, sys, re

budget = float(sys.argv[1]) * 1024 * 1024
metric = sys.argv[sys.argv.index('--metric') + 1] if '--metric' in sys.argv else 'both'
BPW = {'q6_k': 6.5625, 'q5_k': 5.5, 'q4_k': 4.5, 'q3_k': 3.4375, 'iq4_xs': 4.25, 'iq3_s': 3.4375, 'iq3_xxs': 3.0625,
       'iq2_s': 2.5625, 'iq2_xs': 2.3125, 'iq2_xxs': 2.0625, 'iq1_m': 1.75, 'mxfp4': 4.25, 'nvfp4': 4.5, 'q8_0': 8.5}
H, I, V = 2048, 7680, 128000
CONV = [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
ATTN = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

rows = list(csv.reader(open('D:/Models/AMQL/lfm3-work/sens2.csv')))
def cost(label, base='q6_k'):
    for r in rows:
        if r[0] == base and r[1] == label:
            ood, chat = float(r[3]), float(r[5])
            return {'chat': chat, 'ood': ood, 'both': (chat + ood) / 2}[metric]
    return None

base_cost = cost('none')
groups = []
# Per-layer FFNs: measured at q6_k (≈0) and iq2_xxs; interpolate the other
# types by bits (cost ∝ 2^(-2·bpw), a standard MSE-vs-bits model).
def interp(c_low, low_bpw, bpw):
    return c_low * 2 ** (-2 * (bpw - low_bpw))
FINE = '--fine' in sys.argv
if '--factor' in sys.argv:
    # cost(layer, tensor) ≈ cost(layer's whole FFN) × tensor's share, the
    # shares measured with every layer's gate / up / down lowered at once.
    share = {t: max(cost(f'ffn_{t}') - base_cost, 1e-5) for t in ('gate', 'up', 'down')}
    total = sum(share.values())
    for l in range(24):
        d = max(cost(f'ffnL{l}') - base_cost, 1e-5)
        for t in ('gate', 'up', 'down'):
            dt = d * share[t] / total
            # ffnL was measured at iq2_xxs; the group shares at iq2_s.
            opts = {q: interp(dt, BPW['iq2_xxs'], BPW[q]) for q in ['q6_k', 'q5_k', 'q4_k', 'iq3_s', 'iq3_xxs', 'iq2_s', 'iq2_xs', 'iq2_xxs']}
            opts['q6_k'] = 0.0
            groups.append((rf'blk\.{l}\.ffn_{t}\.weight', H * I, opts))
for l in range(24) if '--factor' not in sys.argv else []:
    parts = [('gate', f'fgateL{l}'), ('up', f'fupL{l}'), ('down', f'fdownL{l}')] if FINE else [('(gate|up|down)', f'ffnL{l}')]
    for tname, label in parts:
        c = cost(label)
        if c is None:
            continue
        d = max(c - base_cost, 1e-5)
        opts = {t: interp(d, BPW['iq2_xxs'], BPW[t]) for t in ['q6_k', 'q5_k', 'q4_k', 'iq3_s', 'iq3_xxs', 'iq2_s', 'iq2_xs', 'iq2_xxs']}
        opts['q6_k'] = 0.0
        groups.append((rf'blk\.{l}\.ffn_{tname}\.weight', (1 if FINE else 3) * H * I, opts))
# Mixers and the token interface: measured at iq2_s (mixers) / per type (embedding).
def mixer(label, regex, n):
    c = cost(label)
    d = max(c - base_cost, 1e-5)
    opts = {t: interp(d, BPW['iq2_s'], BPW[t]) for t in ['q6_k', 'q5_k', 'q4_k', 'iq3_s', 'iq2_s']}
    opts['q6_k'] = 0.0
    groups.append((regex, n, opts))
mixer('conv_in', r'blk\.\d+\.shortconv\.in_proj\.weight', len(CONV) * 3 * H * H)
mixer('conv_out', r'blk\.\d+\.shortconv\.out_proj\.weight', len(CONV) * H * H)
mixer('attn_qkv', r'blk\.\d+\.attn_[qkv]\.weight', len(ATTN) * (H * H + 2 * 512 * H))
mixer('attn_out', r'blk\.\d+\.attn_output\.weight', len(ATTN) * H * H)
emb = {'q6_k': 0.0}
for t, lab in [('q5_k', 'emb_q5k'), ('q4_k', 'emb_q4k'), ('nvfp4', 'emb_nvfp4'), ('mxfp4', 'emb_mxfp4')]:
    emb[t] = max(cost(lab) - base_cost, 0.0)
groups.append((r'token_embd\.weight', V * H, emb))

choice = [min(g[2], key=lambda t: g[2][t]) for g in groups]
size = lambda: sum(g[1] * BPW[c] / 8 for g, c in zip(groups, choice)) + 8 * 1024 * 1024  # + F32 norms etc.
while size() > budget:
    best = None
    for i, (rx, n, opts) in enumerate(groups):
        cur = choice[i]
        for t, c in opts.items():
            saved = n * (BPW[cur] - BPW[t]) / 8
            if saved <= 0:
                continue
            ratio = (c - opts[cur]) / saved
            if best is None or ratio < best[0]:
                best = (ratio, i, t)
    if best is None:
        break
    choice[best[1]] = best[2]
pred = base_cost + sum(g[2][c] for g, c in zip(groups, choice))
print(f'# allocate.py budget {sys.argv[1]} MB, metric {metric}: predicted size {size() / 1048576:.0f} MB, predicted KLD {pred:.4f}')
for (rx, n, opts), c in zip(groups, choice):
    print(f'{rx} {c}')
print(r'blk\.\d+\.attn_gate\.weight q8_0')
print('* q6_k')
