"""Minimal float32 LFM3 text reference, written from modeling_lfm3.py.

usage: lfm3_ref.py <model_dir> <comma-separated token ids> [--dump out.json]
Prints per-layer residual L2 norms of the last row and the top-5 next tokens.
"""
import json, sys, math
import torch
import torch.nn.functional as F
from safetensors import safe_open

model_dir = sys.argv[1]
ids = [int(t) for t in sys.argv[2].split(',')]
dump = sys.argv[sys.argv.index('--dump') + 1] if '--dump' in sys.argv else None
dev = 'cuda'

cfg = json.load(open(f'{model_dir}/config.json'))
tc = cfg.get('text_config', cfg)
H = tc['hidden_size']; nh = tc['num_attention_heads']; nkv = tc['num_key_value_heads']; hd = tc['head_dim']
eps = tc['norm_eps']; types = tc['layer_types']; win = tc['sliding_window']
sandwich = tc.get('use_sandwich_norm', False)

import glob
W = {}
for f in glob.glob(f'{model_dir}/*.safetensors'):
    with safe_open(f, 'pt') as sf:
        for k in sf.keys():
            if k.startswith('model.vision_tower') or k.startswith('model.multi_modal_projector'):
                continue
            n = k.replace('model.language_model.', '').replace('model.', '', 1) if k.startswith('model.') else k
            W[n] = sf.get_tensor(k).to(dev, torch.float32)

def rms(x, w):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

def inv_freq(lt):
    rp = tc['rope_parameters'][lt]
    base = rp['rope_theta']; pf = rp.get('partial_rotary_factor', 1.0)
    angles = int(pf * hd) // 2
    f = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32, device=dev) / hd))
    return torch.cat([f, torch.zeros(hd // 2 - angles, device=dev)])

def rot_half(x):
    a, b = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-b, a], -1)

T = len(ids)
pos = torch.arange(T, device=dev, dtype=torch.float32)
x = W['embed_tokens.weight'][torch.tensor(ids, device=dev)]
norms = []
for l, lt in enumerate(types):
    p = f'layers.{l}.'
    h = rms(x, W[p + 'operator_norm.weight'])
    if lt == 'conv':
        bcx = h @ W[p + 'conv.in_proj.weight'].T  # [T, 3H]
        B, C, xx = bcx.split(H, -1)
        bx = (B * xx).T.unsqueeze(0)  # [1,H,T]
        w = W[p + 'conv.conv.weight']  # [H,1,K]
        K = w.shape[-1]
        y = F.conv1d(bx, w, padding=K - 1, groups=H)[..., :T][0].T
        out = (C * y) @ W[p + 'conv.out_proj.weight'].T
    else:
        q = (h @ W[p + 'self_attn.q_proj.weight'].T).view(T, nh, hd)
        k = (h @ W[p + 'self_attn.k_proj.weight'].T).view(T, nkv, hd)
        v = (h @ W[p + 'self_attn.v_proj.weight'].T).view(T, nkv, hd)
        q = rms(q, W[p + 'self_attn.q_norm.weight']); k = rms(k, W[p + 'self_attn.k_norm.weight'])
        fr = pos[:, None] * inv_freq(lt)[None]
        emb = torch.cat([fr, fr], -1)
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]
        q = q * cos + rot_half(q) * sin; k = k * cos + rot_half(k) * sin
        k = k.repeat_interleave(nh // nkv, 1); v = v.repeat_interleave(nh // nkv, 1)
        s = torch.einsum('qhd,khd->hqk', q, k) / math.sqrt(hd)
        qi = torch.arange(T, device=dev)[:, None]; ki = torch.arange(T, device=dev)[None]
        mask = ki > qi
        if lt == 'sliding_attention':
            mask = mask | (qi - ki >= win)
        s = s.masked_fill(mask[None], float('-inf')).softmax(-1)
        a = torch.einsum('hqk,khd->qhd', s, v)
        g = torch.sigmoid(h @ W[p + 'self_attn.gate_proj.weight'].T)[..., None]
        out = (a * g).reshape(T, nh * hd) @ W[p + 'self_attn.o_proj.weight'].T
    if sandwich:
        out = rms(out, W[p + 'post_operator_norm.weight'])
    x = x + out
    f_in = rms(x, W[p + 'ffn_norm.weight'])
    ff = (F.silu(f_in @ W[p + 'feed_forward.w1.weight'].T) * (f_in @ W[p + 'feed_forward.w3.weight'].T)) @ W[p + 'feed_forward.w2.weight'].T
    if sandwich:
        ff = rms(ff, W[p + 'post_ffn_norm.weight'])
    x = x + ff
    norms.append(x[-1].norm().item())
    print(f'layer {l:2d} {lt:18s} |r|={norms[-1]:.4f}')

xf = rms(x, W['embedding_norm.weight'])
head = W.get('lm_head.weight', W['embed_tokens.weight'])
logits = xf @ head.T
top = logits[-1].topk(5)
print('top5', [(i.item(), round(v.item(), 4)) for v, i in zip(top.values, top.indices)])
if dump:
    json.dump({'ids': ids, 'norms': norms, 'last_logits_top': [[i.item(), v.item()] for v, i in zip(top.values, top.indices)],
               'all_top1': logits.argmax(-1).tolist(), 'logits': logits.tolist()}, open(dump, 'w'))
