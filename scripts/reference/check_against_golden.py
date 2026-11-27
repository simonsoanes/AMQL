"""Checks stream_reference.py against make_arch_zoo.py's goldens: the streamed
model must give transformers' own whole-model logits. Usage:
  python -P check_against_golden.py <fixture dir> <reference out dir>
"""
import json, sys
import numpy as np

fixture, out = sys.argv[1], sys.argv[2]
golden = json.load(open(f"{fixture}/golden.json"))
summary = json.load(open(f"{out}/summary.json"))
prompt = summary["prompts"][0]
assert prompt["ids"] == golden["ids"], (prompt["ids"], golden["ids"])
want = np.array(golden["logits"], dtype=np.float32)
got = np.fromfile(f"{out}/{prompt['name']}.all.f32", dtype="<f4").reshape(len(golden["ids"]), -1)[:, : want.shape[1]]
worst = float(np.abs(got - want).max())
print(f"{fixture.rsplit('/', 1)[-1]}: worst |diff| {worst:.3g}, argmax agree {int((got.argmax(1) == want.argmax(1)).sum())}/{len(want)}, "
      f"{summary['layer_loads']} layer loads")
sys.exit(0 if worst < 1e-3 else 1)
