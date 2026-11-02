"""Norm-tweak healing for a low-bit LFM3 GGUF.

The student is the quantized GGUF, dequantized and frozen; only its F32
tensors — every RMSNorm weight (incl. the sandwich post-norms, which act as
per-channel gains on each sub-block's output) and the short-conv kernels —
are trained to minimise KL(teacher || student) against the BF16 teacher GGUF
on calibration text. The tuned tensors are written back into a copy of the
quantized GGUF in place (same F32 shapes), so the result loads anywhere the
original did.

usage: heal_norms.py <teacher.gguf> <student.gguf> <out.gguf> <calib.txt>[,<calib2.txt>...]
       [--steps N] [--seq T] [--batch B] [--lr LR] [--train conv,norms]
"""
import math, shutil, sys, time, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, 'D:/Dev/Tools/llama.cpp/gguf-py')
import gguf
from gguf.quants import dequantize
from tokenizers import Tokenizer

args = sys.argv[1:]
teacher_path, student_path, out_path, calib = args[:4]
def opt(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default
STEPS, SEQ, BATCH, LR = opt('--steps', 300), opt('--seq', 512), opt('--batch', 4), opt('--lr', 2e-3)
TRAIN = opt('--train', 'norms,conv').split(',')
dev = 'cuda'
torch.manual_seed(0); random.seed(0)

def kv(r, key):
    f = r.fields[key]
    if f.types[0] == gguf.GGUFValueType.ARRAY:
        return [f.parts[i].tolist()[0] for i in f.data]
    return f.parts[-1].tolist()[0]

def load(path, dtype):
    r = gguf.GGUFReader(path)
    W = {}
    for t in r.tensors:
        a = dequantize(t.data, t.tensor_type)
        shape = list(reversed([int(x) for x in t.shape]))
        W[t.name] = torch.from_numpy(np.ascontiguousarray(a.reshape(shape), dtype=np.float32)).to(dev, dtype)
    return r, W

tr, TW = load(teacher_path, torch.bfloat16)
sr, SW = load(student_path, torch.bfloat16)
A = 'lfm3'
n_layer = kv(tr, f'{A}.block_count'); H = kv(tr, f'{A}.embedding_length')
nh = kv(tr, f'{A}.attention.head_count'); kvh = kv(tr, f'{A}.attention.head_count_kv')
hd = kv(tr, f'{A}.attention.key_length'); eps = kv(tr, f'{A}.attention.layer_norm_rms_epsilon')
swa = kv(tr, f'{A}.attention.sliding_window_pattern'); win = kv(tr, f'{A}.attention.sliding_window')
base_full = kv(tr, f'{A}.rope.freq_base'); base_swa = kv(tr, f'{A}.rope.freq_base_swa')
conv_layer = [h == 0 for h in kvh]
rope_mask = TW['rope_freqs.weight'].float()

# Trainable log-gains on the student's F32 tensors: w = w0 * exp(s), so every
# step is a relative change whatever the tensor's scale (plain Adam on w moves
# small norm weights by huge relative amounts).
base = {}
logs = {}
for name, t in SW.items():
    is_norm = name.endswith('norm.weight')
    is_conv = name.endswith('shortconv.conv.weight')
    if (is_norm and 'norms' in TRAIN) or (is_conv and 'conv' in TRAIN):
        base[name] = t.float().clone()
        logs[name] = torch.nn.Parameter(torch.zeros_like(base[name]))

class Gains(dict):
    def __getitem__(self, k):
        return base[k] * torch.exp(logs[k])
    def __contains__(self, k):
        return k in base
params = Gains()
print(f'trainable tensors: {len(logs)} ({sum(p.numel() for p in logs.values())} values)', flush=True)

def inv_freq(full):
    base = base_full if full else base_swa
    f = 1.0 / (base ** (torch.arange(0, hd, 2, device=dev, dtype=torch.float32) / hd))
    return f / rope_mask if full else f

INV = {True: inv_freq(True), False: inv_freq(False)}

def rms(x, w):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(torch.bfloat16)

def rot(x, cos, sin):
    a, b = x[..., : hd // 2], x[..., hd // 2:]
    xr = torch.cat([-b, a], -1)
    return (x.float() * cos + xr.float() * sin).to(x.dtype)

def get(Wd, P, name):
    return P[name] if P is not None and name in P else Wd[name]

def layer(x, l, Wd, P, cos_sin, mask):
    p = f'blk.{l}.'
    B, T, _ = x.shape
    h = rms(x, get(Wd, P, p + 'attn_norm.weight'))
    if conv_layer[l]:
        bcx = h @ Wd[p + 'shortconv.in_proj.weight'].T
        Bg, Cg, xx = bcx.split(H, -1)
        bx = (Bg * xx).transpose(1, 2)
        w = get(Wd, P, p + 'shortconv.conv.weight')  # [H, K]
        K = w.shape[-1]
        y = F.conv1d(F.pad(bx.float(), (K - 1, 0)), w.float().unsqueeze(1), groups=H).to(torch.bfloat16)
        out = (Cg * y.transpose(1, 2)) @ Wd[p + 'shortconv.out_proj.weight'].T
    else:
        full = not swa[l]
        cos, sin = cos_sin[full]
        q = (h @ Wd[p + 'attn_q.weight'].T).view(B, T, nh, hd)
        k = (h @ Wd[p + 'attn_k.weight'].T).view(B, T, kvh[l], hd)
        v = (h @ Wd[p + 'attn_v.weight'].T).view(B, T, kvh[l], hd)
        q = rms(q, get(Wd, P, p + 'attn_q_norm.weight')); k = rms(k, get(Wd, P, p + 'attn_k_norm.weight'))
        q, k = rot(q, cos, sin), rot(k, cos, sin)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[full], enable_gqa=True)
        a = a.transpose(1, 2)  # B T nh hd
        g = torch.sigmoid((h @ Wd[p + 'attn_gate.weight'].T).float()).to(a.dtype).unsqueeze(-1)
        out = (a * g).reshape(B, T, nh * hd) @ Wd[p + 'attn_output.weight'].T
    if p + 'post_attention_norm.weight' in Wd:
        out = rms(out, get(Wd, P, p + 'post_attention_norm.weight'))
    x = x + out
    f_in = rms(x, get(Wd, P, p + 'ffn_norm.weight'))
    ff = (F.silu(f_in @ Wd[p + 'ffn_gate.weight'].T) * (f_in @ Wd[p + 'ffn_up.weight'].T)) @ Wd[p + 'ffn_down.weight'].T
    if p + 'post_ffw_norm.weight' in Wd:
        ff = rms(ff, get(Wd, P, p + 'post_ffw_norm.weight'))
    return x + ff

def forward(ids, Wd, P, grad):
    B, T = ids.shape
    pos = torch.arange(T, device=dev, dtype=torch.float32)
    cs = {}
    for full in (True, False):
        fr = pos[:, None] * INV[full][None]
        emb = torch.cat([fr, fr], -1)
        cs[full] = (emb.cos()[None, :, None], emb.sin()[None, :, None])
    qi = torch.arange(T, device=dev)[:, None]; ki = torch.arange(T, device=dev)[None]
    mask = {True: ki <= qi, False: (ki <= qi) & (qi - ki < win)}
    x = Wd['token_embd.weight'][ids]
    for l in range(n_layer):
        if grad:
            x = checkpoint(layer, x, l, Wd, P, cs, mask, use_reentrant=False)
        else:
            x = layer(x, l, Wd, P, cs, mask)
    x = rms(x, get(Wd, P, 'output_norm.weight'))
    head = Wd.get('output.weight', Wd['token_embd.weight'])
    return x, head

def kl_loss(xs, hs, xt, ht, chunk=512):
    # KL(teacher || student) over the vocabulary, chunked over tokens.
    xs = xs.reshape(-1, xs.shape[-1]); xt = xt.reshape(-1, xt.shape[-1])
    total = 0.0
    n = xs.shape[0]
    for i in range(0, n, chunk):
        ls = (xs[i:i + chunk] @ hs.T).float()
        with torch.no_grad():
            lt = (xt[i:i + chunk] @ ht.T).float()
            pt = F.log_softmax(lt, -1)
        total = total + F.kl_div(F.log_softmax(ls, -1), pt, log_target=True, reduction='sum')
    return total / n

tok = Tokenizer.from_file('D:/Models/LFM3-VL-2B/tokenizer.json')
ids = []
for path in calib.split(','):
    text = open(path, encoding='utf-8').read()
    ids += tok.encode(text, add_special_tokens=False).ids
# Last 10% of the calibration tokens validate (early stopping); the rest train.
split = int(len(ids) * 0.9)
data = torch.tensor(ids[:split], dtype=torch.long)
val = torch.tensor(ids[split:], dtype=torch.long)
print(f'calibration tokens: {len(ids)}', flush=True)

def batch():
    starts = torch.randint(0, len(data) - SEQ - 1, (BATCH,))
    return torch.stack([data[s:s + SEQ] for s in starts]).to(dev)

optim = torch.optim.AdamW(logs.values(), lr=LR, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * s / STEPS)))
fixed = [torch.stack([val[s:s + SEQ] for s in range(i * BATCH * SEQ, (i + 1) * BATCH * SEQ, SEQ)]).to(dev)
         for i in range(max(1, min(4, (len(val) - 1) // (BATCH * SEQ))))]

def evaluate():
    with torch.no_grad():
        tot = 0.0
        for b in fixed:
            xs, hs = forward(b, SW, params, False); xt, ht = forward(b, TW, None, False)
            tot += kl_loss(xs, hs, xt, ht).item()
        return tot / len(fixed)

best_kl = evaluate()
best = {k: v.detach().clone() for k, v in logs.items()}
print(f'step 0 val KL {best_kl:.4f}', flush=True)
t0 = time.time()
for step in range(1, STEPS + 1):
    b = batch()
    with torch.no_grad():
        xt, ht = forward(b, TW, None, False)
    xs, hs = forward(b, SW, params, True)
    loss = kl_loss(xs, hs, xt, ht)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(logs.values(), 1.0)
    optim.step(); sched.step()
    if step % 25 == 0 or step == STEPS:
        v = evaluate()
        if v < best_kl:
            best_kl = v
            best = {k: t.detach().clone() for k, t in logs.items()}
        print(f'step {step} loss {loss.item():.4f} val KL {v:.4f} (best {best_kl:.4f}, {time.time() - t0:.0f}s)', flush=True)
with torch.no_grad():
    for k in logs:
        logs[k].copy_(best[k])

# Write the tuned F32 tensors into a copy of the student GGUF, in place.
shutil.copyfile(student_path, out_path)
r = gguf.GGUFReader(student_path)
offsets = {t.name: (t.data_offset, t.tensor_type, t.n_bytes) for t in r.tensors}
with open(out_path, 'r+b') as f:
    for name in base:
        p = params[name]
        off, ttype, nbytes = offsets[name]
        assert ttype == gguf.GGMLQuantizationType.F32, name
        data = p.detach().float().cpu().numpy().astype(np.float32).tobytes()
        assert len(data) == nbytes, (name, len(data), nbytes)
        f.seek(off); f.write(data)
print('wrote', out_path)
