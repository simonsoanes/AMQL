"""Ablate the merged checkpoint: map merged embedding rows back onto LFM3's own
token ids (dropping appended imported-only rows), optionally reverting the blend.

usage: merge_ablate.py <merged-export-dir> <lfm3-dir> <out-dir> [--alpha A]
alpha = weight of the merged (blended) row; 0 reverts to LFM3, 1 keeps the merge."""
import json, shutil, sys, os
import torch
from safetensors.torch import load_file, save_file

merged_dir, lfm_dir, out_dir = sys.argv[1:4]
alpha = float(sys.argv[sys.argv.index('--alpha') + 1]) if '--alpha' in sys.argv else 1.0
os.makedirs(out_dir, exist_ok=True)
mt = json.load(open(f'{merged_dir}/tokenizer.json', encoding='utf-8'))
lt = json.load(open(f'{lfm_dir}/tokenizer.json', encoding='utf-8'))
mvocab = dict(mt['model']['vocab']); mvocab.update({a['content']: a['id'] for a in mt['added_tokens']})
lvocab = dict(lt['model']['vocab']); lvocab.update({a['content']: a['id'] for a in lt['added_tokens']})

M = load_file(f'{merged_dir}/model.safetensors')
L = load_file(f'{lfm_dir}/model.safetensors')
key = 'model.language_model.embed_tokens.weight'
orig = L[key].float()
emb = orig.clone()
hit = 0
for tok, lid in lvocab.items():
    mid = mvocab.get(tok)
    if mid is not None and lid < emb.shape[0]:
        emb[lid] = alpha * M[key][mid].float() + (1 - alpha) * orig[lid]
        hit += 1
print('mapped rows', hit, 'changed rows', int((emb != orig).any(-1).sum()))
out = dict(L)
out[key] = emb.to(L[key].dtype)
save_file(out, f'{out_dir}/model.safetensors', metadata={'format': 'pt'})
for f in os.listdir(lfm_dir):
    if f.endswith('.json') or f.endswith('.jinja'):
        shutil.copy(f'{lfm_dir}/{f}', f'{out_dir}/{f}')
