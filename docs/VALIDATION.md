# Implemented evidence and remaining checks

The recorded results were obtained with CPU-bounded offline replay, synthetic
fixtures and localhost transport. No GPU training, controller deployment, robot
integration or motion was performed.

## Passed software checks

| Check | Evidence |
|---|---|
| Python focused suite excluding socket tests | 145 passed |
| Localhost transport/restart/failure/exact-wire checks | 4 passed, ephemeral 127.0.0.1 only |
| Pinned Java owner prediction checks | 109 assertions passed, compiled into this task |
| Cross-language golden vectors |Java emitted byte-identical request/reply; Python schema/semantic consumption passed |
| Abstention cross-check |Java/Python masked reply semantics equal; Java gate `UNCALIBRATED`, baseline retained |
| Clean upstream reference integrity | 77 extracted files matched recorded SHA-256, including locally preserved reference assets |
| Source/core parity | 15 tests cover exact source bytes/import relocation, architecture, indexing/residual, gradients, default sampled loss, auxiliary outputs, carry correction, reset/isolation and stochastic/native forward parity |
| Package |Local wheel built; exact schemas/core BSD license included; no ANYmal weights, actor/PPO/simulator/logger code in wheel; extracted wheel loaded its contracts and frozen tensors |
| Minimal analytical dependency path |Extracted wheel imported analytical/service on bare Python 3.12 without torch/numpy |
| Tiny learning and reload | 2 CPU updates, batch 4, five GRU heads; frozen reload output parity passed; validation/calibration plumbing, no final-test tuning |

Owned-file whitespace checks pass; clean upstream copies retain their original whitespace for exact source parity.

The reproduced upstream defect was shared recurrent carry leaking between
training member rollouts. The correction resets the appropriate shared base at
member entry. It is named `RWM-U-source/carry-isolated-v1`, not unchanged upstream
behavior. Independent freshly reset head tensor/loss parity passes. The active
source loss remains sampled MSE with bound regularization; bootstrap defaults
false. Gaussian NLL and full MBPO bootstrap are documented alternatives, not
silently selected.

Other fixed defects demonstrated during review: overlapping request provenance
had erased identical physical snapshots as conflicts; conversion now merges
contexts and detects actual conflicting facts. Evaluation hashes formerly bound
only window IDs; they now bind actual results/layout/failures. Final-test
thresholds formerly accepted arbitrary checksums/negative tolerances; they now
bind model/dataset/layout/split/calibration/horizon and compare measured test
metrics to frozen limits. Regression checks pass.

The suite covers source/epoch/session/candidate/batch isolation, units/frame and
causal alignment, angle normalization, reset/gaps/duplicates/out-of-order logs,
unknown/masked state, crash/restart/disconnect/timeout, invalid schema/NaN/clock/
frame, corrupt/path-invalid package, stale/old replies, cancellation and overload.
Missing learned predictions preserve the analytical replay baseline. No tests
establish physical stopping, object pickup success or hardware safety.

## Measured CPU plumbing

`evidence/native-cpu-benchmark.json` is the exact Mac ARM64/Python 3.12.14/
PyTorch 2.8.0 report: one CPU thread, validated synthetic frozen weights,
history 32, five members, two candidates, eight forecast steps, 3 warm-ups and 30
samples. Timings include serialized worker queue and JSON encoding/decoding.

| Milliseconds |p50|p95|p99/max|
|---|---:|---:|---:|
|Queue|0.01496|0.01792|0.01833|
|Native inference|17.55958|18.31513|18.51588|
|Serialization|0.41675|0.45675|0.46900|
|End-to-end|17.99017|18.76542|18.93446|

There were 0 misses against the synthetic 20 ms budget; process peak RSS was
213,860,352 bytes (~204 MiB). These are observations of this small sequential
Mac test, not guarantees or qualified deployment limits. Power, thermal and
simultaneous vision load are unknown/unrun. No controller or Jetson was used.

`evidence/synthetic-smoke.json` records hashes and training/validation counts.
Six validation windows succeeded for RWM, persistence, constant velocity,
analytical response, history ablation and action ablation in each reported mode.
Synthetic labels and toy constants establish plumbing only; physical prediction
error and candidate ranking on FRC are unknown.

## Unrun and blocking prerequisites

No real authorized FRC logs, independent labels, pickup outcomes, robot footprint,
stopping/delay/mechanism tolerances, physical covariance calibration or held-out
real sessions were supplied. V1 also lacks mechanism state/output and clipping/
saturation/execution-deviation telemetry. Eight-state learned mechanism inference
cannot be populated from v1. A matched ego6/body3 package and complete causal
accepted-command history are required before useful raw wire shadow diagnostics.
All wire replies currently abstain, preserving external analytical forecasts.

Final-test metrics, pickup precision/recall/Brier/calibration, trustworthy
counterfactual ranking, Jetson stack inventory/native latency/power/thermal/vision
coexistence and hardware failures are unrun. Large training, Dreamer/PETS,
ONNX/TensorRT/quantization, robot integration and deployment are unimplemented or
on hold. No hardware qualification is claimed.

WPILib 2026+roboRIO is the immediate ecosystem and 2027 alpha-7+Systemcore is a
separate experimental profile; this component is controller independent. The
[official SystemcoreTesting matrix](https://github.com/wpilibsuite/SystemcoreTesting#software-compatibility)
inspected 2026-10-08 lists 2027 alphas only. No official supported WPILib 2026+
Systemcore target was found. No compatibility layer or HAL rewrite is invented.
JSON `_us` remains microseconds independent of NT version metadata.

Next smallest step: review the local source/contract/package and obtain an
explicitly authorized short causal log with complete accepted chassis commands,
source/capture/sync/session identities and known labels, with predictions having
no command influence. Real training/calibration/device experiments require their
own approved scope. Live advisory promotion needs held-out, on-device and shadow
evidence plus explicit approval; robot integration stays on hold.

## Publication portability verification

A clean local clone of the reviewed `main` snapshot passed all 149 Python tests
and built the wheel using the previously installed, pinned CPU test environment.
No other component checkout was needed for ordinary Python setup/tests/build.
Fresh dependency installation on another host is unrun. The optional Java check
passed with the explicit byte-verified checkout override. Public history excludes
local coordination/permission notes and private filesystem paths; the original
development history is retained only locally. `docs/evidence/package.json`
records the sanitized wheel checksum. No model weights or datasets are published.

The optional Java check was subsequently verified against a fresh public
World-State clone at `0e5b6c3f85d47205cde8b0c80e13fa79d35e95ab`: all five
Java source hashes and both schema/fixture pairs matched, 109 Java assertions
passed, and Java emission/Python consumption plus masked baseline parity passed.
The public export manifest hash also matched. This is a metadata-only handoff;
model/runtime/schema bytes and hardware limits are unchanged. Evidence is in
`evidence/world-state-public-verification.json`.
