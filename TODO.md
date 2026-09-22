# TODO — issues noticed while building the WPF front-end (Amql.Gui)

Raised 2026-09-22 during construction of `src/Amql.Gui` (AMQL Studio).
**All eight items are now fixed** (kept here as the record of what changed and
where; see `CHANGELOG.md`).

## 1. No machine-readable progress on stdout — FIXED

Long-running commands (`encode`, `export`, `moe-ify`, `prune`, `fine-tune`,
`generate-mtp --fit`, `import`) printed human-readable lines only, so the GUI
had to scrape `NN%` / `N/M` patterns out of the output, which could misfire on
unrelated numbers.

**Fix:** `src/Amql.Cli/CliProgress.cs` implements the proposed protocol, on
under the new global `--progress` flag (or `AMQL_PROGRESS=1`/`true`). One JSON
object per line, newline-terminated and flushed immediately:

```
##amql-progress {"command":"route","phase":"probe","done":3,"total":8,"percent":37.5}
##amql-result   {"command":"path","found":false,"nodes":48,"budgetNodes":48}
```

`Phase` / `Advance` / `Complete` drive the progress records (wired into
`encode`, `route` and `path` so far), `Result` carries the machine-readable
outcome. Documented in `amql-cli help`. The GUI passes `--progress` and binds to
it directly, keeping `ParseProgress` only as a fallback for an older CLI.

## 2. Stack traces on stderr — FIXED

`Program.Main` caught a generic `Exception` and printed `e.ToString()` (full
stack trace) with exit 2.

**Fix:** a failure is now one `error: <message>` line, plus the hint
`re-run with --verbose for the full stack trace`; the new global `--verbose`
flag restores the full trace. Typed `CliException` / `MergeException` failures
were already single-line and are unchanged apart from the exit-code constant.

## 3. `change-tensor` reported one error per run — FIXED

The edit op/value, the three positionals and the mandatory `--out` were
validated first-thing-throws, so a user fixed one argument per invocation.

**Fix:** all validation problems are collected into a list and thrown as one
`CliException` listing every one of them (`  - missing …` lines).

## 4. `verify` hardcoded the Qwen3.5 operand probes — FIXED

`Program.Verify` resolved `3.self_attn.q_proj.weight`,
`0.linear_attn.in_proj_qkv.weight`, `0.linear_attn.A_log` and
`target.final_norm/weight` by name. On containers without linear-attention
layers — or with fewer than four, e.g. the `synth-model` demo — those
resolutions threw and `verify` exited non-zero even though integrity passed.

**Fix:** each probe is wrapped; a tensor that cannot be resolved prints
`[skipped — <reason>]` and the run continues. The closing line reports how many
probes matched, or states that no shape-specific probe matched this container
(integrity above is unaffected).

## 5. `route` progress wrote without newlines — FIXED

`RelationRouter.Route(…, Console.Write, …)` streamed partial progress text, so
under redirected stdout (how the GUI runs it) updates that never ended a
newline stayed invisible until process exit.

**Fix:** route and path now pass `CliProgress.FragmentWriter`. Attached to a
console it still writes inline (the `···` display is unchanged); when output is
redirected each fragment is newline-terminated and flushed, so every update is
visible as it happens. `Main` ends any pending fragment line before returning.

## 6. Exit codes conflated "failed" and "negative result" — FIXED

`path` returned 1 when no path was found within the budget — a legitimate
answer, not an error — and `verify` returned 1 on an integrity failure, both
indistinguishable from usage errors for a front-end.

**Fix:** the codes are now named constants with a documented table in the help
text and in `Program.cs`:

| Code | Meaning |
|------|---------|
| `0` `ExitOk` | success — the command produced its answer, whatever it was |
| `1` `ExitNegativeResult` | a legitimate negative result (`path` found no chain within budget; `verify`'s integrity check failed). A `##amql-result` line carries the reason when `--progress` is on |
| `2` `ExitUsage` | usage or runtime error — bad arguments, unreadable input, an unsupported operator, an exception |

`amql-cli help` with no arguments returns 2 (usage); `amql-cli help <cmd>` returns 0.
The GUI records exit 1 as `Negative` rather than `Failed`, and the progress bar
treats it as a completed run.

## 7. `to-gguf` had no `--force` — FIXED

**Fix:** `--force` deletes an existing output file and proceeds; without it the
error now says `pass --force to overwrite`. Re-run pipelines are idempotent.

## 8. Pre-existing build warnings in `Amql.Gguf` — FIXED

`HfCheckpointToGguf.cs` had eight CS8602/CS8604 warnings (lines 424, 436, 440,
506, 600, 641, 694, 722) from dereferencing the nullable `PlanEntry.Source`.

**Fix:** `PlanEntry` gained a `RequiredSource` property that throws a typed
`GgufException` naming the transform and tensor when the plan is malformed, and
every call site uses it. The solution now builds warning-clean.

---

### GUI-side follow-ups (optional, no core change needed)

- Protocol emission is wired into `encode`, `route` and `path`; the other
  long-running commands (`export`, `moe-ify`, `prune`, `fine-tune`,
  `generate-mtp --fit`, `import`) still rely on the GUI's heuristic fallback.
  Adding `CliProgress.Advance` calls inside their loops would make them exact.
- `CliRunner.ParseProgress` stays as the fallback for a CLI that predates
  `--progress`; it can be removed once every distributed build supports it.
