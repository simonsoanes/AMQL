"""Runs an ONNX file written by `amql-cli export --format onnx` in onnxruntime
and compares its logits with a golden.json made by transformers
(tests/Amql.Tests/fixtures/make_arch_zoo.py).

    python check_onnx.py model.onnx golden.json
"""
import json, sys
import numpy as np
import onnxruntime as ort

path, golden_path = sys.argv[1], sys.argv[2]
g = json.load(open(golden_path))
ids = np.array([g["ids"]], dtype=np.int64)
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
logits = session.run(["logits"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0][0]
worst = float(np.abs(logits - np.array(g["logits"])).max())
print(f"worst logit diff vs transformers {worst:.3g}")
sys.exit(0 if worst < 2e-3 else 1)
