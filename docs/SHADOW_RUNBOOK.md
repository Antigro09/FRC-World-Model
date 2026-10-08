# Offline and shadow runbook

Current mode is reviewable offline/shadow code. Every exact v1 wire reply remains
masked abstention; raw model diagnostics cannot influence commands. No robot
project, controller topic, deployed service, interlock, motor or CAN interface
is included. Robot integration remains on hold.

## Review and replay

1. Review `PROVENANCE.md`, exact `contracts/world-state-v1` hashes, `BINDINGS.md`
   and method `RWM-U-source/carry-isolated-v1`. Review every changed preprocessing,
   carry, orchestration and action-index behavior. Preserve the clean references.
2. Use an isolated Python 3.12 CPU environment and the resolved lock. Run
   `.venv/bin/python -m pytest -q`. Loopback tests bind ephemeral `127.0.0.1`
   only. No controller or external endpoint is contacted.
3. Replay a copied golden request using `tools/replay_wire.py`, supplying an
   explicit microsecond sync bound and new output path. Validate the resulting
   reply using the exact schema. Expected outcome is full masked abstention and
   baseline selection. The fixture is synthetic and has no adequate model history.
4. For cross-language checks, run `tools/verify_cross_language.py` against the
   owner checkout with an exact `--source-ref COMMIT_SHA` or the explicit hash-verified `--working-tree` override, and a new output directory. The tool reads pinned
   Java source and writes all classes/vectors into the supplied output directory; it never edits the owner.
5. Preserve authorized original logs. Supply the converter's external session,
   capture, label, causal-input and complete-command-history metadata. Reject
   unavailable facts and configured mechanism inputs that ego-only would discard.
   Inspect hashes, exclusions, resets, gaps, duplicates and clock mapping first.

## Offline experiment scope

The synthetic generator and bounded trainer are CPU-only. They are plumbing
checks, not a robot accuracy study. Use new output directories and train-only
normalization. Split whole connected sessions/days before windows into train,
validation, calibration and final test. The tiny trainer permits at most 100
updates / batch 32; that cap does not authorize a large real-data experiment.
A real run requires a reviewed dataset, device and experiment scope.

Compare persistence, constant velocity and analytical response on the same
windows and horizons. Calibrate analytical constants using designated data;
synthetic response constants are not robot calibration. Evaluate one-step,
open-loop multistep and refreshed-cutoff runs separately. Future measured states
are labels only. Inspect history/action ablations, per-session/tail errors,
missing forecasts and provenance; keep sensor error, stochastic spread,
ensemble disagreement and empirical forecast error separate.

Before final-test inspection, obtain physical tolerances from actual footprint,
stopping, latency and mechanism measurements. `tools/freeze_thresholds.py`
binds user-supplied positive tolerances and session-max descriptive bounds to the
frozen calibration report, model, dataset, layouts, split and horizon. The final
evaluator verifies the artifact checksum/provenance and reports the policy and
observed tolerance/session-bound comparisons. No current threshold artifact is
robot approved. Quantiles are descriptive; they do not prove safety or coverage.

## Frozen inference and optional work

A frozen package has CPU tensor weights, SHA-256 for every artifact, resolved
source/configuration, exact schema/layout, train-only normalization, dataset and
calibration metadata. `frozen.py` loads tensors with `weights_only=True` on CPU;
it does not import a trainer, download weights, upgrade a stack or select CUDA.
`FrozenPackage.load` rejects corrupted, escaping/symlinked or oversized artifacts.
Inference extras and training extras are separate. Verify the wheel's exact
schema resources and core license; ANYmal assets, PPO, actors and simulator
reference code are excluded from the wheel.

`FrozenWireShadow` accepts an explicitly supplied mapped robot-clock callback,
model/package identity and complete history/configuration prerequisites. It
requires matching ego6/body3 layout, fixed timestep and complete valid history.
A nonblocking lock sheds competing optional calls. Native rollouts reset carry
for each member and candidate, retain source/cutoff/epoch in diagnostics, and
never become fresh state. Missing inputs, exceptions and overload yield masked
abstention. Expired requests fail; the caller keeps its analytical baseline.

For local transport checks, `transport.loopback_server(handler)` binds
`127.0.0.1` on an ephemeral port, uses one serialized worker, a two-connection
listen queue, bounded 1 MB NDJSON frames and <=2s timeouts. It creates no production
endpoint. The consumer must validate every response against its original
request/current epoch and discard old, malformed, stale or uncalibrated replies.

The separate private `PredictionService` harness tests cancellation/reset and
bounded pending work. Its synthetic 8-state envelope and internal statuses are
not the production contract. Uncooperative callbacks cannot be forcibly stopped;
late replies are discarded, and deployment would require bounded native work.

## Hardware evidence still required

No deployment is authorized by this runbook. Before any later device work,
separately inventory actual JetPack, ARM64, CUDA, Python and PyTorch plus the
working perception load, without upgrading it. Benchmark faithful native frozen
inference first: warm-up, queue, serialization, candidates/horizons, p50/p95/p99/
max, deadline misses, memory, power, thermal behavior and simultaneous vision.
Set explicit request budgets and shed optional predictions before perception
starvation. The Mac synthetic benchmark is not a Jetson qualification.

ONNX/TensorRT/quantization remain optional unimplemented optimizations. Each would
need tensor, carry, autoregressive rollout, uncertainty and decision parity.
No current package relies on any optimization, cloud service or weight download.

Promotion to live advisory use requires held-out real-data evidence, calibrated
physical covariance and abstention, on-device coexistence/latency evidence,
shadow logs proving no command influence, and explicit approval.
Executed-action logs alone do not establish counterfactual candidate ranking or
closed-loop safety. Tracking/planning/interlocks and authoritative robot facts
remain owned by their respective components. Actual robot integration is on hold.
