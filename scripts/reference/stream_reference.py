"""Builds a transformers reference for a model of any size in a few GB of memory.

transformers is the model authors' own arithmetic, so it is the authority AMQL's
runtimes are checked against. Loading a large model whole (and its quantised
experts dequantised) does not fit in memory, so this keeps the model on the meta
device and gives each decoder layer its weights only while it runs: they are
read from the checkpoint and converted by transformers' own loader (the same
renames and dequantisation from_pretrained applies), used, and freed. A small
cache keeps layers that run again soon (a looped core, the next decode step).

What is written is only what a comparison needs, gathered as the model runs:

  summary.json      config facts, prompts and their ids, per position the top-k
                    logits, the log-sum-exp and entropy, and a greedy continuation
  visits.json       per layer visit (each time a decoder layer runs, in order) and
                    position: RMS, mean, max |x| and the first coordinates of the
                    visit's output, plus a fixed 16-direction projection
                    ("fingerprint") that a cosine compares
  aggregates.json   running statistics per visit over every prompt and position
                    (Welford mean and variance of the RMS, max |x|), and, for any
                    module that returns integer indices (an expert router), how
                    often each index was chosen
  <name>.last.f32   the last position's full logits, float32 little-endian

transformers is used here only, for setup; nothing in AMQL depends on it.

Usage (run at low priority; see the repo's notes on reserving CPU):
  python -P stream_reference.py <model dir> <out dir> [--prompt TEXT]... [--chat TEXT]...
         [--dtype bfloat16|float32] [--threads 8] [--topk 20] [--generate 8]
         [--cache-gb 3] [--full-logits]
"""
import argparse, glob, json, math, os, sys, time
from collections import OrderedDict

# Convert on the calling thread: the loader's worker pool keeps materialised
# tensors alive past their conversion, and memory then grows with each layer.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming, convert_and_load_state_dict_in_model, rename_source_key
from transformers.modeling_utils import LoadStateDictConfig


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def _memory():
    """(working set, peak working set, private bytes) in GiB."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        return info.rss / 2**30, getattr(info, "peak_wset", info.rss) / 2**30, getattr(info, "private", info.rss) / 2**30
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        psapi = ctypes.WinDLL("psapi")
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.WorkingSetSize / 2**30, c.PeakWorkingSetSize / 2**30, c.PrivateUsage / 2**30
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        return peak, peak, peak
    except ImportError:
        return float("nan"), float("nan"), float("nan")


def rss_gb():
    return _memory()[0]


def peak_gb():
    return _memory()[1]


# ── the model on the meta device ─────────────────────────────────────────────

def build_meta_model(model_dir, dtype):
    config = AutoConfig.from_pretrained(model_dir)
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(text, dtype=dtype)
    model.eval()
    return model, text


def materialise_missing(model, provided):
    """What the checkpoint does not hold (buffers computed at construction,
    such as rotary frequencies, and any parameter it lacks) gets real storage
    once and is filled by the model's own init, as from_pretrained does for its
    missing keys. These are never released."""
    owners = {}
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if name in provided or value.device.type != "meta":
            continue
        owner, _, attr = name.rpartition(".")
        module = model.get_submodule(owner) if owner else model
        if attr in module._parameters:
            module._parameters[attr] = nn.Parameter(torch.empty_like(value, device="cpu"), requires_grad=False)
        else:
            module._buffers[attr] = torch.empty_like(value, device="cpu")
        owners[owner] = module
    with torch.no_grad():
        for module in owners.values():
            model._init_weights(module)
    return sorted(owners)


def find_stack(model):
    """The ModuleList holding most of the parameters: the decoder layers."""
    best, best_count = None, -1
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > 1:
            count = sum(p.numel() for p in module.parameters())
            if count > best_count:
                best, best_count = name, count
    if best is None:
        raise SystemExit("no layer stack found")
    return best


# ── weights on demand ────────────────────────────────────────────────────────

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
           "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
           "BOOL": torch.bool, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2}


class FileTensors:
    """A safetensors file read a tensor at a time with plain file reads.
    safe_open maps the whole file copy-on-write, and Windows charges such a
    mapping in full against the commit limit (19 GB for Inkling-2-Mini before
    a byte is read): this holds only the tensors being converted."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        self.base = 8 + n
        header.pop("__metadata__", None)
        self.entries = header

    def keys(self):
        return self.entries.keys()

    def get_slice(self, key):
        return _LazyTensor(self, key)

    def read(self, key):
        e = self.entries[key]
        start, end = e["data_offsets"]
        buf = bytearray(end - start)
        with open(self.path, "rb", buffering=0) as f:
            f.seek(self.base + start)
            view, done = memoryview(buf), 0
            while done < len(buf):
                got = f.readinto(view[done:])
                if not got:
                    raise IOError(f"{self.path}: short read of {key}")
                done += got
        dtype = _DTYPES[e["dtype"]]
        if len(buf) == 0:
            return torch.empty(e["shape"], dtype=dtype)
        return torch.frombuffer(buf, dtype=dtype).reshape(e["shape"])


class _LazyTensor:
    """What the loader is given for a checkpoint tensor: `t[...]` reads it."""

    def __init__(self, file, key):
        self.file, self.key = file, key
        e = file.entries[key]
        self.shape = torch.Size(e["shape"])
        self.dtype = _DTYPES[e["dtype"]]

    def get_shape(self):
        return list(self.shape)

    def get_dtype(self):
        return self.file.entries[self.key]["dtype"]

    def __getitem__(self, index):
        t = self.file.read(self.key)
        return t if index is Ellipsis else t[index]

class LazyWeights:
    """Maps every checkpoint tensor to the model parameter transformers loads it
    into, and loads the parameters under a module prefix on request."""

    def __init__(self, model, model_dir, dtype):
        self.model = model
        self.dtype = dtype
        index = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(index):
            files = sorted(set(json.load(open(index))["weight_map"].values()))
        else:
            files = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))]
            files = [f for f in files if f == "model.safetensors" or f.startswith("model-")] or files
        self.handles = {f: FileTensors(os.path.join(model_dir, f)) for f in files}
        self.mapping = get_model_conversion_mapping(model, None, None)
        renamings = [m for m in self.mapping if isinstance(m, WeightRenaming)]
        converters = [m for m in self.mapping if isinstance(m, WeightConverter)]
        meta = model.state_dict()
        self.targets = set(meta)
        self.sources = {}    # target (first) parameter -> [(file, key)]
        unmatched = 0
        for f, h in self.handles.items():
            for key in h.keys():
                renamed, _ = rename_source_key(key, renamings, converters, model.base_model_prefix, meta)
                if renamed not in meta and key in meta:
                    renamed = key
                if renamed in meta:
                    self.sources.setdefault(renamed, []).append((f, key))
                else:
                    unmatched += 1
        self.dtype_plan = model._get_dtype_plan(dtype) if hasattr(model, "_get_dtype_plan") else {}
        log(f"weights: {sum(len(v) for v in self.sources.values())} checkpoint tensors for "
            f"{len(self.sources)} parameters, {unmatched} unused (other modules or towers)")

    def under(self, prefix):
        return [t for t in self.sources if t.startswith(prefix)]

    def load(self, targets):
        state = {}
        for t in targets:
            for f, key in self.sources[t]:
                state[key] = self.handles[f].get_slice(key)
        config = LoadStateDictConfig(device_map={"": "cpu"}, dtype=self.dtype, dtype_plan=self.dtype_plan,
                                     weight_mapping=self.mapping)
        info, _ = convert_and_load_state_dict_in_model(self.model, state, config)
        if info.conversion_errors:
            raise SystemExit(f"conversion errors: {info.conversion_errors}")

    def release(self, prefix):
        """Puts every loaded parameter and buffer under the prefix back on the meta device."""
        for name, module in self.model.named_modules():
            if not (name == prefix or name.startswith(prefix + ".")):
                continue
            for pname, p in list(module._parameters.items()):
                full = f"{name}.{pname}" if name else pname
                if p is not None and p.device.type != "meta" and full in self.sources:
                    module._parameters[pname] = nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=False)
            for bname, b in list(module._buffers.items()):
                full = f"{name}.{bname}" if name else bname
                if b is not None and b.device.type != "meta" and full in self.sources:
                    module._buffers[bname] = torch.empty_like(b, device="meta")


class LayerCache:
    """Gives a layer its weights just before it runs; keeps the most recently
    used layers up to a budget of bytes, and frees the rest."""

    def __init__(self, weights, budget_bytes):
        self.weights = weights
        self.budget = budget_bytes
        self.resident = OrderedDict()   # prefix -> bytes
        self.largest = 0                # the largest layer seen: what the next may need
        self.loads = 0

    def ensure(self, prefix):
        if prefix in self.resident:
            self.resident.move_to_end(prefix)
            return
        # Make room first, so the layers kept and the one arriving fit the
        # budget together (one layer always loads, whatever the budget).
        while self.resident and sum(self.resident.values()) + self.largest > self.budget:
            old, _ = self.resident.popitem(last=False)
            self.weights.release(old)
        targets = self.weights.under(prefix + ".")
        t0 = time.time()
        self.weights.load(targets)
        size = sum(p.numel() * p.element_size() for n, p in self.weights.model.get_submodule(prefix).named_parameters())
        self.resident[prefix] = size
        self.largest = max(self.largest, size)
        self.loads += 1
        log(f"  load {prefix}: {size / 2**30:.2f} GiB in {time.time() - t0:.1f}s, "
            f"{len(self.resident)} resident, rss {rss_gb():.1f} GiB")


# ── what is gathered ─────────────────────────────────────────────────────────

class Welford:
    def __init__(self):
        self.n, self.mean, self.m2, self.max = 0, 0.0, 0.0, 0.0

    def add(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        self.max = max(self.max, x)

    def json(self):
        return {"n": self.n, "mean": self.mean, "std": math.sqrt(self.m2 / self.n) if self.n > 1 else 0.0, "max": self.max}


class Recorder:
    FIRST = 8
    DIRECTIONS = 16

    def __init__(self, hidden):
        self.hidden = hidden
        self._directions = {}
        self.visit = 0
        self.prompt = None
        self.visits = {}       # prompt -> list of per-visit records
        self.aggregates = {}   # visit -> {"rms": Welford, "absmax": Welford}
        self.choices = {}      # module name -> {index: count}
        self.active = True

    def directions(self, width):
        """Fixed random unit directions (seeded by the width): a cosine over
        these compares two hidden states cheaply without storing them. A layer
        may carry several residual streams side by side (Qwen4: width a
        multiple of the hidden size), so they are made for each width seen."""
        if width not in self._directions:
            g = torch.Generator().manual_seed(20261127 + width)
            self._directions[width] = torch.nn.functional.normalize(
                torch.randn(self.DIRECTIONS, width, generator=g), dim=1)
        return self._directions[width]

    def start(self, prompt):
        self.prompt = prompt
        self.visit = 0
        self.visits[prompt] = []

    def on_layer(self, layer_name, output):
        if not self.active:
            return
        h = output[0] if isinstance(output, (tuple, list)) else output
        h = h.detach().float()[0]                      # [positions, hidden]
        rms = h.pow(2).mean(-1).sqrt()
        absmax = h.abs().amax(-1)
        record = {
            "layer": layer_name,
            "rms": [round(v, 6) for v in rms.tolist()],
            "mean": [round(v, 6) for v in h.mean(-1).tolist()],
            "absmax": [round(v, 6) for v in absmax.tolist()],
            "first": [[round(v, 6) for v in row] for row in h[:, : self.FIRST].tolist()],
            "width": h.shape[-1],
            "fingerprint": [[round(v, 6) for v in row] for row in (h @ self.directions(h.shape[-1]).T).tolist()],
        }
        self.visits[self.prompt].append(record)
        agg = self.aggregates.setdefault(self.visit, {"layer": layer_name, "rms": Welford(), "absmax": Welford()})
        for r, a in zip(rms.tolist(), absmax.tolist()):
            agg["rms"].add(r)
            agg["absmax"].add(a)
        self.visit += 1

    def on_indices(self, name, output):
        if not self.active:
            return
        items = output if isinstance(output, (tuple, list)) else (output,)
        for t in items:
            if torch.is_tensor(t) and not t.is_floating_point() and t.dtype != torch.bool and t.numel() > 0:
                counts = self.choices.setdefault(name, {})
                for v, c in zip(*torch.unique(t, return_counts=True)):
                    counts[int(v)] = counts.get(int(v), 0) + int(c)


# ── running it ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--prompt", action="append", default=[])
    ap.add_argument("--chat", action="append", default=[])
    ap.add_argument("--ids", action="append", default=[], help="comma-separated token ids")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--generate", type=int, default=8)
    ap.add_argument("--cache-gb", type=float, default=3.0)
    ap.add_argument("--full-logits", action="store_true", help="every position's full logits (small vocabularies)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    model, text_config = build_meta_model(args.model, dtype)
    weights = LazyWeights(model, args.model, dtype)
    initialised = materialise_missing(model, set(weights.sources))
    if initialised:
        log(f"not in the checkpoint, initialised by the model: {', '.join(initialised[:6])}{' …' if len(initialised) > 6 else ''}")
    stack = find_stack(model)
    layers = model.get_submodule(stack)
    cache = LayerCache(weights, int(args.cache_gb * 2**30))

    # Everything outside the stack (embedding, final norm, head, the loop's
    # mixing) is loaded once and kept.
    resident = [t for t in weights.sources if not t.startswith(stack + ".")]
    weights.load(resident)
    if getattr(text_config, "tie_word_embeddings", False):
        model.tie_weights()
    missing = [n for n, p in model.named_parameters() if p.device.type == "meta" and not n.startswith(stack + ".")]
    if missing:
        raise SystemExit(f"parameters outside the stack not in the checkpoint: {missing[:5]}")
    log(f"model on meta, {len(layers)} layers in '{stack}', resident set loaded, rss {rss_gb():.1f} GiB, {time.time() - t0:.0f}s")

    hidden = text_config.hidden_size
    recorder = Recorder(hidden)
    for i, layer in enumerate(layers):
        prefix = f"{stack}.{i}"
        layer.register_forward_pre_hook(lambda m, a, p=prefix: cache.ensure(p))
        layer.register_forward_hook(lambda m, a, o, p=prefix: recorder.on_layer(p, o))
    for name, module in model.named_modules():
        if name.endswith(("router", "gate")) and not isinstance(module, nn.Linear):
            module.register_forward_hook(lambda m, a, o, n=name: recorder.on_indices(n, o))

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:  # a test fixture without a tokenizer gives ids
        log(f"no tokenizer: {e}")
    prompts = []
    for i, p in enumerate(args.prompt):
        prompts.append((f"plain{i}", p, tokenizer(p, return_tensors="pt").input_ids[0].tolist()))
    for i, c in enumerate(args.chat):
        text = tokenizer.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True)
        prompts.append((f"chat{i}", text, tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()))
    for i, s in enumerate(args.ids):
        prompts.append((f"ids{i}", None, [int(x) for x in s.split(",")]))
    if not prompts:
        raise SystemExit("give at least one --prompt, --chat or --ids")

    vocab = getattr(text_config, "unpadded_vocab_size", None) or text_config.vocab_size
    summary = {"model": os.path.abspath(args.model), "model_type": text_config.model_type, "dtype": args.dtype,
               "hidden_size": hidden, "layers": len(layers), "stack": stack, "vocab": vocab, "prompts": []}
    for name, text, ids in prompts:
        t1 = time.time()
        recorder.start(name)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()[:, :vocab]
        log(f"{name}: {len(ids)} tokens, {recorder.visit} layer visits, {time.time() - t1:.0f}s, rss {rss_gb():.1f} GiB")
        positions = []
        lse = torch.logsumexp(logits, -1)
        probs = torch.softmax(logits, -1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum(-1)
        top = torch.topk(logits, min(args.topk, vocab), dim=-1)
        for t in range(len(ids)):
            positions.append({"top_ids": top.indices[t].tolist(), "top_logits": [round(v, 5) for v in top.values[t].tolist()],
                              "logsumexp": round(lse[t].item(), 5), "entropy": round(entropy[t].item(), 5)})
        logits[-1].numpy().astype("<f4").tofile(os.path.join(args.out, f"{name}.last.f32"))
        if args.full_logits:
            logits.numpy().astype("<f4").tofile(os.path.join(args.out, f"{name}.all.f32"))

        # A greedy continuation, decoded with the model's own cache; the layer
        # statistics are for the prompt pass alone.
        recorder.active = False
        greedy = []
        if args.generate > 0:
            with torch.no_grad():
                out = model.generate(input_ids=torch.tensor([ids]), max_new_tokens=args.generate, do_sample=False,
                                     use_cache=True)
            greedy = out[0, len(ids):].tolist()
        recorder.active = True
        summary["prompts"].append({"name": name, "text": text, "ids": ids, "positions": positions, "greedy": greedy,
                                   "greedy_text": tokenizer.decode(greedy) if tokenizer and greedy else None})
        log(f"{name}: greedy {greedy} {repr(tokenizer.decode(greedy)) if tokenizer and greedy else ''}, {time.time() - t1:.0f}s total")

    summary["layer_loads"] = cache.loads
    summary["seconds"] = round(time.time() - t0, 1)
    summary["peak_working_set_gib"] = round(peak_gb(), 2)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), ensure_ascii=False)
    json.dump(recorder.visits, open(os.path.join(args.out, "visits.json"), "w"))
    json.dump({"visits": {str(k): {"layer": v["layer"], "rms": v["rms"].json(), "absmax": v["absmax"].json()}
                          for k, v in recorder.aggregates.items()},
               "choices": {k: dict(sorted(v.items())) for k, v in recorder.choices.items()}},
              open(os.path.join(args.out, "aggregates.json"), "w"))
    log(f"done in {summary['seconds']}s, {cache.loads} layer loads, peak working set {peak_gb():.2f} GiB")


if __name__ == "__main__":
    main()
