# FRC-World-Model

Frozen structured-state, action-conditioned prediction for **offline replay and
shadow review**. Robot-code integration, deployment and motion remain
on hold. Source: [Antigro09/FRC-World-Model](https://github.com/Antigro09/FRC-World-Model).

The service owns neither tracking nor pose fusion, planning, commands or safety
interlocks. Missing, rejected or uncalibrated learned forecasts preserve the
analytical baseline. No raw video, policy/actor training, online exploration,
simulator or CAN/motor interface is involved.

The implementation reuses actual pinned RWM `SystemDynamicsEnsemble`, `RNNBase`
GRU and state/auxiliary heads. The reference configuration has two GRU layers
with 256 hidden units, 128-unit heads and five members. The shared recurrent
bases are preserved: five heads do not mean five independent full GRUs.
The explicit carry-isolation correction is documented and source-tested.
ANYmal data and weights are reference artifacts, never FRC training weights.

## Reproduce on the approved offline workstation

Python 3.12 is required. The tested environment is macOS ARM64, Python 3.12.14,
PyTorch 2.8.0, one CPU thread. `requirements-cpu-test.lock` records all resolved
test dependencies. Training and inference extras are separate; the analytical
service uses the standard library. Installation creates only a task-local venv.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-cpu-test.lock
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
.venv/bin/python tools/synthetic_smoke.py artifacts/smoke-input
OMP_NUM_THREADS=1 .venv/bin/python tools/train_offline.py artifacts/smoke-input/synthetic.jsonl --config artifacts/smoke-input/config.json --output artifacts/smoke-package --max-updates 2 --batch-size 4
OMP_NUM_THREADS=1 .venv/bin/python tools/evaluate_offline.py artifacts/smoke-package artifacts/smoke-input/synthetic.jsonl --config artifacts/smoke-input/config.json --output artifacts/smoke-validation.json
OMP_NUM_THREADS=1 .venv/bin/python tools/benchmark_rwm.py --package artifacts/smoke-package --output artifacts/native-cpu-benchmark.json --samples 30 --candidates 2
```

Output paths must be new: logs and existing packages are never overwritten.
Tiny synthetic training proves ingestion, gradients, serialization and replay;
it does not prove FRC accuracy, calibrated covariance or candidate ranking.
The trainer caps this path at 100 updates and batch size 32. Larger authorized
experiments require a separately reviewed dataset/device/experiment scope.
Final-test metrics require a previously frozen calibration threshold artifact
with explicit physical tolerances. No final-test results have been used to tune
this component.

## Contracts and limits

FRC-World-State owns `frc-prediction/1`; its schemas and golden vectors are copied
unchanged into `contracts/world-state-v1`. JSON `*_us` remains microseconds.
Internal local-log helpers use explicitly named nanoseconds; conversions must
be exact. NT timestamp metadata belongs in version-specific controller adapters.
No production topic is defined here.

V1 wire forecasts contain position and field velocity only. History has ego
heading/velocity and accepted command provenance, but no mechanism state,
complete dataset session/day identity, execution deviation or clipping log.
The explicit ego-only input adapter uses six ego features and three commands.
The initial eight-state/four-action ego/mechanism representation is a **pending
offline local-log envelope**, not an alternative production contract. It cannot
be populated from v1 by substituting zero for missing state. Missing real
prerequisites and missing approved calibration cause abstention.

Consult `docs/PROVENANCE.md`, `docs/DATA.md`, `docs/BINDINGS.md`,
`docs/TARGETS.md`, `docs/SHADOW_RUNBOOK.md` and `docs/VALIDATION.md` for the
source mapping, deviations, replay contract and remaining evidence gates.

No real authorized FRC logs, independent labels, pickup outcomes, robot
footprint/stopping/delay/mechanism tolerances, approved calibration sessions or
Jetson inventory were supplied. Consequently forecasts and ranking remain
advisory. External object tracks remain analytical; pickup heads are disabled.

The next smallest step is review of the code and exact exported contract, then
an authorized short robot-owned log with accepted commands and complete causal
state/provenance, recorded without prediction influencing commands.

The Python package and test suite are self-contained. Optional Java contract
verification needs JDK17 and a separately supplied
[World-State checkout](https://github.com/Antigro09/FRC-World-State), with an exact
`--source-ref` commit or explicit hash-verified `--working-tree` override; see
`docs/BINDINGS.md`. No other checkout is silently required by setup/build.
