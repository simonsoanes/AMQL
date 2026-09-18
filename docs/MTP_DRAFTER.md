# Pre-trained MTP drafter for AMQL — an engineered, calculation-first design

**Goal:** turn a dense container that ships no MTP (e.g. `Qwen3.5-0.8B`) into one that
carries a *pre-trained* MTP drafter — where "pre-trained" means the drafter's free
parameters are **calculated**, not optimised by stochastic gradient descent: one forward
pass over a corpus collects the model's own continuation map, and a closed-form ridge
least-squares solve replaces training. No autograd, no SGD, no random seeds in the
learning path.

The known baseline it must beat: the bootstrapped `generate-mtp` drafter (trunk = a copy
of the last full-attention layer, projector `fc` = the mean-combination boot) measured at
**0.3% zero-shot draft acceptance** on `Qwen3.5-0.8B`. Every phase in this design gates on
the same deterministic acceptance measurement.

## Principles

1. **Calculation over training.** The drafter's free block is fit by a ridge-regularised
   least-squares projection (order-independent, closed-form). "Trial and error" is limited
   to choosing a small set of *scalar* hyper-parameters (ridge, clusters, features) — each
   chosen by a principled default and confirmed by the measurement gate, never by search.
2. **Determinism.** Corpus → pairs → weights → acceptance is a pure function of (container,
   corpus, scalars). Same inputs, byte-identical weights; validated by a determinism test.
3. **Reuse.** The build reuses existing, tested machinery: the ridge solver
   (`Amql.Merge.LeastSquares.Fit/ResidualL2` from the anchored-alignment merge), the
   balanced k-means + materialised linear router (`MoeIfy.ClusterUnits`, the moe-ify
   router construction), the runtime acceptance gate (`GenerateMtp.MeasureDraftAcceptance`),
   and the module materialisation / export-companion path (`GenerateMtp.Transform`,
   `ModelExporter.ExportMtp`).
4. **Measurement-gated.** No phase is "complete" without its acceptance / residual number,
   computed through the runtime path on a **held-out** split.

## The drafter composition being optimised

The pre-trained drafter keeps the deterministic skeleton and replaces only its free block:

```
x_t = concat( pre_fc_norm_hidden(h_t), pre_fc_norm_embedding(e_{t+1}) )   ∈ R^{2h}   (frozen)
p_t = FREE_BLOCK(x_t)                                                       ∈ R^h      (the fitted part)
y_t = trunk(p_t)                                                            ∈ R^h      (Rms → causal GQA + fused σ-gate → silu FFN → residual; FROZEN, copied from the last full-attention layer)
logits_t = head( mtp_norm(y_t) )                                            ∈ R^vocab  (FROZEN, the shared head)
draft_t = argmax logits_t                                                  → token t+2
```

`generate-mtp` already assembles this skeleton and materialises it. The design below
defines how `FREE_BLOCK` is calculated so the composition maximises acceptance.

## The data contract (one forward pass, deterministic)

The frozen model's own continuation map is collected over a corpus (the acceptance gate's
forward pass, extended to export pairs):

| tensor | size per position | definition |
|---|---|---|
| `x_t` | `2h` | the drafter's input: normed hidden `h_t` (post-final-norm) and the normed embedding `e_{t+1}` |
| `y_t` | `h` | the regression target: the model's **pre-final-norm residual hidden state at t+1** (the autograd-free proxy for "what the trunk input should produce next") |
| `token_t+2` | 1 | the held-out acceptance target (used only by the gate, never by the fit) |

**Held-out split is enforced:** fit on positions `[0, N−M)`, gate on the last `M` (default
`M = 1024`, corpus `N ≥ N_fit + M`). The acceptance number is never measured on fitted
positions.

## The core calculation: ridge least-squares projector

The free block in its first, exact form is a single linear map
`p_t = W·x_t`, `W ∈ R^{h × 2h}`, fit to the collected pairs:

```
W* = argmin_W  Σ_t ‖ y_t − W·x_t ‖²  +  λ · ‖W‖²_F
W* = (XᵀX + λI)⁻¹ XᵀY                  (rows: positions; X ∈ R^{n×2h}, Y ∈ R^{n×h})
```

- This is exactly the system `Amql.Merge.LeastSquares.Fit(a, b, n, d, ridge)` already
  solves in the anchored-alignment merge (with `a = Y`, `b = X`), parallel, double
  precision, ridge floor for degenerate sets.
- **Why this beats SGD here:** the drafter's dependency on its free block is *linear at
  fixed trunk/head*; the L2 objective is convex with a closed-form optimum. SG stumbles on
  a ten-thousand-parameter learning-rate dance; a ridge solve is exact, order-independent
  and reproduces byte-for-byte.
- **Defaults, chosen by principle, confirmed by the gate:** `λ` relative `1e-4` (the split
  used by the merge), `W` stored in the container's stack dtype (BF16/F32 per the segment
  convention the export already handles).
- **Diagnostic, not search:** report the ridge-solve residual `R²` and the eigenspectrum
  tail of `XᵀX` (conditioning of a `2h × 2h` Gram at real scale is the risk to watch; the
  ridge floor keeps it solvable).

## The refinement ladder (K-projector mixture)

If the single linear block underperforms (its ceiling is real: one autoregressive step is
not globally linear), the **engineered** refinement is a small mixture of linear experts —
reusing the MoE machinery already built for `moe-ify`:

1. cluster the collected `x_t` column-vectors into `K` groups with
   `MoeIfy.ClusterUnits`-style balanced k-means (fixed seed — deterministic);
2. fit **one ridge projectors per cluster** on its members: `W_k = (X_kᵀX_k + λI)⁻¹ X_kᵀY_k`;
3. route with the moe-ify linear router: `r_k = mean of the cluster's x-vectors`;
   `p_t = W_{argmax r·x_t}·x_t` (hard top-1 routing; each expert is its own segment:
   separable, inspectable);
4. materialise as the drafter's free block with an `mtp`-facts block the export already
   carries, so re-encoding retains the routed structure (the retention work done for MoE
   applies unchanged).

`K` defaults 8; the gate measures `K ∈ {1, 4, 8}`; deterministic, no search.

## Components

| Component | New/Reused | Role |
|---|---|---|
| `PairCollector` | new (~60 lines, extends the gate's forward) | one dense forward → `x_t`, `y_t`, held-out split on disk/stream |
| `RidgeProjector` | new (~40 lines over `LeastSquares.Fit`) | `W* = (XᵀX+λI)⁻¹XᵀY`, plus `R²`/spectrum diagnostics |
| `KProjectors` | new (~60 lines) | balanced clustering + per-cluster ridge fits + router rows |
| `DrafterAssembly` | new (~50 lines) | writes the fitted block into the `mtp.stack` skeleton (replaces `fc`) |
| `GenerateMtp.MeasureDraftAcceptance` | reused | the runtime acceptance gate (held-out) |
| `GenerateMtp.Transform` / `ModelExporter.ExportMtp` | reused | materialisation + export companion |
| `MoeIfy.ClusterUnits` + router | reused | the K-projector refinement |

## Phases

### Phase 0 — the data contract (no model change)
- Extend the acceptance-gate forward into a `PairCollector`: one pass emits
  `(x_t, y_t, token_{t+2})` over `N` positions with the held-out split.
- **Acceptance criteria:** `x_t` shape `[n_fit, 2h]`, `y_t` `[n_fit, h]`; determinism test
  (two runs → byte-identical pairs); a tiny synthetic container round-trips the sink.

### Phase 1 — the single ridge projector
- `RidgeProjector`: fit `W` on the fit split; assemble the drafter with `fc := W`
  (shape `[h, 2h]`, same dtype path as the boot).
- Materialise + export companion via the existing paths; report `R²`, spectrum tail, and
  held-out acceptance through the runtime gate.
- **Acceptance criteria (gate):** held-out acceptance measurably above the 0.3% boot on
  the real 0.8B (e.g. ≥ 5%), with the weight bytes reproduced across two identical runs.

### Phase 2 — the K-projector mixture
- `KProjectors` + router; `K ∈ {1, 4, 8}` measured on the same held-out split; the
  acceptance gate decides the shipped default `K` (no search beyond the three candidates).
- **Acceptance criteria:** acceptance non-decreasing with `K` (monotone up to noise);
  each expert separable (own segment) and the routed structure retained across
  export → re-encode (the MoE retention test).

### Phase 3 — hardening and integration
- `fit-mtp` CLI surface (`amql-cli generate-mtp --fit [--clusters K] [--ridge 1e-4]`),
  help text, README section with the measured numbers.
- Determinism + held-out + retention tests in the suite; `verify` passes on the fitted
  container; export emits the companion automatically.
- **Acceptance criteria:** full suite green; the fitted container plans/serves; the README
  states the measured `before → after` acceptance with the split semantics.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Linear map ceiling (one step is nonlinear) | the K-projector ladder; measured per `K`, not assumed |
| Ill-conditioned Gram at `2h` scale | ridge + floor (already in the solver); spectrum diagnostic gates the phase |
| Corpus distribution skew — acceptance is domain-local | held-out split and honest reporting (fit vs gate split documented); larger corpus = better, exactly like any drafter |
| The trunk copy may mismatch the reference's trunk (e.g. qk-norm/rope subtleties) | the forward-equivalence discipline stays: the fitted block is verified against the runtime path by construction (the gate *is* the runtime path) |
| Feature choice (concat halves + norms) | principled default fixed in the data contract; alternates are a scalar switch, measured, not tuned |

## Open decisions (set by measurement, each with a principled default)

1. `λ` — relative ridge, default `1e-4` (the merge's value).
2. `K` — shipped default after `{1, 4, 8}`; start 8.
3. `M` (held-out size) — 1024; `N_fit` floor — the CLI enforces `≥ chunk + M`.
4. Storage dtype of `W`/experts — the stack's canonical dtype (segment convention).

## CLI / UX outline (Phase 3)

```
amql-cli generate-mtp <container> --out <out> --fit
                [--clusters 8] [--ridge 1e-4] [--sample 4096] [--eval 1024] [--seed 0]
```

`--fit` runs the calculation path (phases 0–2) instead of the boot-alone path; the report
prints `R²`, the gram spectrum, and `draft acceptance (held-out): boot 0.3% → fitted X%`.