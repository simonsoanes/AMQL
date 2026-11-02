"""Can Qwen3.5-0.8B's embedding space be linearly mapped onto LFM3's on shared tokens?"""
import json, torch
from safetensors import safe_open

def load(path, name):
    with safe_open(path, 'pt') as f:
        return f.get_tensor(name).float().cuda()

q = load('D:/Models/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors', 'model.language_model.embed_tokens.weight')
l = load('D:/Models/LFM3-VL-2B/model.safetensors', 'model.language_model.embed_tokens.weight')
qv = json.load(open('D:/Models/Qwen3.5-0.8B/tokenizer.json', encoding='utf-8'))['model']['vocab']
lv = json.load(open('D:/Models/LFM3-VL-2B/tokenizer.json', encoding='utf-8'))['model']['vocab']
shared = [(qv[t], lv[t]) for t in qv if t in lv and qv[t] < q.shape[0] and lv[t] < l.shape[0]]
print('qwen', q.shape, 'lfm3', l.shape, 'shared', len(shared))
g = torch.Generator().manual_seed(0)
perm = torch.randperm(len(shared), generator=g)
idx = torch.tensor(shared)[perm]
tr, te = idx[: int(0.9 * len(idx))], idx[int(0.9 * len(idx)):]
X, Y = q[tr[:, 0].cuda()], l[tr[:, 1].cuda()]
for lam in (1e-4, 1e-2, 1.0):
    A = torch.linalg.solve(X.T @ X + lam * torch.eye(X.shape[1], device='cuda'), X.T @ Y)
    P = q[te[:, 0].cuda()] @ A
    T = l[te[:, 1].cuda()]
    cos = torch.nn.functional.cosine_similarity(P, T, dim=-1)
    print(f'ridge {lam}: held-out cosine mean {cos.mean():.3f}  frac>0.5 {(cos > 0.5).float().mean():.3f}')
# baseline: nearest-neighbour retrieval accuracy among held-out
P = q[te[:2000, 0].cuda()] @ A; T = l[te[:2000, 1].cuda()]
sim = torch.nn.functional.normalize(P, dim=-1) @ torch.nn.functional.normalize(T, dim=-1).T
print('top1 retrieval among 2000 held-out:', (sim.argmax(-1) == torch.arange(2000, device='cuda')).float().mean().item())
