# Offline action and task outcomes

`frc_world_model.decision_learning` adds a separate, bounded offline outcome
pipeline. It validates causal decision records, splits sessions before fitting,
fits a small empirical comparison baseline, and scores factual task success,
utility and risk on held-out records. It uses the Python standard library,
performs zero optimizer updates, and cannot qualify a live learned predictor.

The version `frc-decision-outcome-log/1` is a local dataset envelope. It is not a
new World-State publication or a replacement for `frc-prediction/1`. Task kinds
match the decision interface: `PICKUP`, `SCORE`, `TRANSIT`, `HOLD`. Observed
possession is `EMPTY`, `HELD`, or `UNKNOWN`; observed mechanism readiness is a
boolean or null. World-State remains the owner of observed tracks, Drive the
owner of robot pose, and roboRIO the owner of accepted commands and interlocks.

## What the current architecture can establish

The existing pinned RWM has shared recurrent GRU bases and separate ensemble
heads. Its matched wire adapter supplies six ego features and three
robot-relative chassis commands. Complete accepted command history is required.
The initial eight-state/four-action local configuration includes synthetic
mechanism features, which cannot be reconstructed from v1 wire state. The
trainer explicitly disables extension/contact/termination losses until a
separate real-label adapter exists. Ensemble spread and head standard deviations
are diagnostics, rather than calibrated physical covariance.

That architecture can learn short-horizon action-conditioned ego transitions
when matched real data and calibration exist. It currently has no trained head
for task completion, possession transitions, mechanism readiness, collection
quantity, or scoring success. A goal or waypoint is insufficient as a physical
action: the controller must provide the expected, timed command sequence for
that candidate. Accepted and actually applied commands may differ after
clipping, saturation, interlocks or controller corrections.

Distance, task preconditions, freshness, current possession/readiness, and
planner-approved reachability can support deterministic task ordering now.
Success probability, acquisition/scoring utility, response delay, physical
collision probability, and ranking improvements over unseen alternatives need
real learned evidence. The outcome table introduced here supplies an offline
comparison method and exercises that data path. It supplies no runtime learned
readiness, pickup success guarantee, sensor covariance or physical safety claim.

## Dataset boundary

Each JSONL line is one complete decision record. Unknown fields fail validation;
future labels cannot be inserted into the causal context. Timestamps are exact
integer microseconds in `ROBOT_MONOTONIC_US`, with `BLUE_FIELD` ego/goal position
and robot-relative command velocity. Initial ego order is
`x_m, y_m, heading_rad, vx_body_mps, vy_body_mps, omega_radps`.

| Field | Requirement |
|---|---|
| `schema_version` | `frc-decision-outcome-log/1` |
| `record_id`, `session_id`, `day` | Nonempty bounded IDs; ISO date; a session spanning midnight keeps its identity |
| `identity` | Exact keys below, each a nonempty bounded string |
| `request_id`, `snapshot_id` | Original request identity and nonnegative snapshot integer |
| `candidate` | Candidate/task IDs, supported task kind, field-frame goal xy, unique target IDs, planner path length in meters |
| `cutoff_us`, `horizon_us`, `step_us` | Nonnegative cutoff, positive integral grid; at most 30 seconds and 64 command samples |
| `context` | Causal observed ego, possession/readiness and their timestamps; clearance and match time remaining |
| `commands_created_us` | At or before the cutoff; this binds what the predictor could know then |
| `candidate_commands` | Complete fixed grid decided at cutoff, containing offset, body twist, explicit mechanism commands |
| `executed_commands` | Separate grid of later accepted commands, their causal acceptance times, and optional measured applied body twists |
| `outcome` | Later, same-identity horizon label with observed success, risk event, utility, label source/provenance and utility/risk definition IDs |
| `time_sync_valid`, `time_sync_error_us`, `time_sync_limit_us` | Synchronization valid and measured error no greater than explicit tolerance |

Identity keys are `robot_id`, `map_id`, `frame_id`, `calibration_id`, `source_id`,
`source_epoch`, `reset_epoch`, `geometry_revision`, `localization_revision`,
`configuration_revision`, `sync_mapping_revision`, and `clock_domain`. The
outcome repeats the same identity. A source/reset/map change across the horizon
rejects the record, rather than joining unrelated observations after a reboot.

Context has exactly `input_causal: true`, `observed_us`, `ego_state`,
`possession`, `possession_observed_us`, `mechanism_ready`,
`mechanism_observed_us`, `minimum_clearance_m`, and `time_remaining_s`.
All context timestamps must be at or before cutoff. Clearance is a logged
planner/context feature; this module does not calculate or certify it.

Candidates have exactly `candidate_id`, `task_id`, `task_kind`, `goal_xy_m`,
`target_ids`, and `path_length_m`. Command samples have `offset_us`,
`body_twist: [vx, vy, omega]`, and `mechanisms`. The executed samples add
`accepted_us`, and may add `applied_body_twist`. Offsets start at zero and match
the declared step. An accepted command must be accepted at or before the start
of its interval. Mechanism values are explicit bounded strings, booleans or
finite numbers, with at most 16 named entries.

Outcome fields are `observed_us`, `identity`, `success`, `risk_event`, `utility`,
`provenance`, `label_source_id`, `utility_definition_id`, and
`risk_definition_id`. The outcome timestamp covers the entire forecast horizon.
Success and risk require observed boolean labels; unknown labels never become
false. Supported label provenance is `measured`, `independent_ground_truth`, or
`synthetic_fixture`. A fused estimate alone does not establish task success or
ground truth. The utility and risk definition IDs fix the labeling semantics
across fitting and evaluation; differing definitions cannot be mixed.

`tests/test_decision_learning.py::outcome_fixture` is a complete synthetic
example of this envelope, including identities, both command grids, timestamps
and labels. Its toy utility and risk definitions are examples only.

`load_decision_jsonl(path)` reads at most 16 MiB and 2,048 source records. It
leaves input bytes unchanged, reports the original SHA-256 and semantic dataset
hash, and counts label provenance. Exact duplicates collapse; conflicting
record or decision identities, duplicate JSON keys at any nesting level,
nonfinite JSON constants, and any malformed record reject the dataset with
a line-specific error. The parser reads at most the byte limit plus one before
decoding JSON, so growing input files cannot bypass the byte limit. The record
and command bounds also apply to Python APIs, including duplicate-only streams.

## Training and evaluation

```python
from frc_world_model.data import SplitConfig
from frc_world_model.decision_learning import (
    load_decision_jsonl, split_decision_sessions,
    fit_outcome_baseline, evaluate_outcomes, OutcomePrediction,
)

records, ingestion = load_decision_jsonl("authorized-outcomes.jsonl")
splits = split_decision_sessions(records, SplitConfig(seed="experiment-1"))
model = fit_outcome_baseline(splits.records["train"], split="train")

# This placeholder is a named comparison example. Supply the actual frozen
# deterministic method from the experiment rather than claiming these values
# describe a robot.
def example_reference(inputs):
    return OutcomePrediction(0., .5, .5, provenance="example_reference")

validation = evaluate_outcomes(
    splits.records["validation"],
    {"empirical_table": model.predict, "example_reference": example_reference},
    split="validation", trained=model,
)
```

The default split groups complete days and joins days connected by a session
crossing midnight. Seeded group ordering and four-way largest-remainder
allocation are deterministic. Small datasets can leave a held-out split empty;
the count remains zero and no score is invented. Explicit session grouping is
available, but evaluation still rejects train-day overlap. Fit accepts only
`split="train"`. Validation, calibration and final-test evaluation reject any
training session, day or record identity.

The empirical table conditions on stable robot/map/configuration/source domain,
task kind, known possession/readiness, horizon, one-meter distance bins,
one-m/s mean command-speed bins, one-rad/s mean turn-rate bins, and the explicit
mechanism command sequence. It uses label means and empirical success/risk
fractions in bins with at least two examples by default. Different commands
therefore can produce different empirical outcomes. Unseen domains/bins,
unknown measured context and unsupported tasks abstain. This coarse table
ignores other continuous details, cannot simulate dynamics, and has no qualified
OOD detector or calibrated uncertainty. Its input bins are documented method
choices rather than robot-validated tolerances.

Labels are excluded from fitting if actual accepted or supplied applied
commands differ from the expected candidate sequence. Deviations remain in the
original validated record. The identical predicate excludes deviated records
from evaluation before invoking any predictor; exclusion counts are reported and
the excluded records reduce coverage. Lack of an applied-command
field is reported by the dataset shape rather than filled with an invented
measurement; accepted commands alone do not prove actuator execution. Label
provenance is retained even when synthetic and measured examples coexist; mixed
provenance does not create real validation evidence.

The immutable `CausalTaskInput` passed to callbacks contains no outcome or later
accepted/applied command records. All callbacks use the same held-out factual
records. Reports provide utility MAE, success Brier, risk Brier, per-task and
per-session scores, prediction coverage, abstention/error counts, per-record
predictions/provenance, and measured callback latency. Errors and abstentions
reduce coverage and never become successful zero-error predictions. The report
hash binds records, actual scores, missing predictions and measured timings.

Executed-action logs do not contain outcomes for alternatives that were never
executed. These reports explicitly withhold counterfactual ranking/regret
claims. Simulator candidate comparisons would require a separately versioned
paired-scene dataset, complete candidate coverage, fixed exogenous state/seed,
and simulator validity evidence. Synthetic task fixtures here do not supply
that evidence or prove closed-loop improvement.

The optional final-test path requires a previously frozen, model-bound policy
from a hashed held-out calibration report:

```python
from frc_world_model.decision_learning import freeze_outcome_test_policy

policy = freeze_outcome_test_policy(
    calibration_report, model_id=model.model_id,
    limits={"utility_mae": approved_utility_limit,
            "success_brier": approved_success_brier_limit,
            "risk_brier": approved_risk_brier_limit,
            "minimum_coverage": approved_minimum_coverage},
)
test = evaluate_outcomes(
    splits.records["test"], frozen_predictors, split="test", trained=model,
    frozen_test_policy=policy,
)
```

The final test cannot reuse calibration sessions or days. Policy hashes,
predictor names and model identity must match, and actual test scores are
compared against frozen limits. Passing such limits remains an offline outcome
score result, with `qualified_for_live: false`.

For a complete bounded preparation/fit/validation/calibration run:

```sh
python -m frc_world_model.decision_learning authorized-outcomes.jsonl \
  --output artifacts/new-outcome-report.json --seed experiment-1
```

The output is created exclusively. Existing logs/reports are never overwritten.
The CLI records split membership, model identity/cells, exclusions, source/data
hashes and validation/calibration scores. Final-test metrics remain uncomputed
and final-test labels are not used for fitting or tuning.

## Bounded local software evidence

On 2026-10-13, `.venv/bin/python -m pytest -q
tests/test_decision_learning.py` passed 38 checks in 0.16 seconds. These cover
strict schemas, future-command/context leakage, unknown labels, nonfinite values,
source/reset boundaries, command grid/deviation checks, input bounds, duplicates,
midnight/day/session isolation, action-conditioned fitting, unsupported-context
abstention, unavailable predictors, factual baseline comparison, frozen final
test policies, and exclusive CLI output.

A separate local run used the same `outcome_fixture` generator with 512 training
records on one day and 128 validation records on a separate day. Even-numbered
records had mean command speed 2 m/s, utility 8 and success true; odd-numbered
records had speed 0, utility 1 and success false. Labels came entirely from that
toy rule. The fitted table and explicit deterministic toy rule both achieved
zero utility MAE, success Brier and risk Brier on all 128 validation records,
with 100% coverage. This verifies factual conditioning and identical scoring;
it supplies no evidence of improved ranking or robot accuracy.

| Measured item | Result |
|---|---|
| Host/runtime | macOS 26.6.2 ARM64, Python 3.12.14 |
| Model identity | `offline-task-table-f17af57f1e5ac85c` |
| Fitted bins / optimizer updates | 2 / 0 |
| Fit wall time, 512 records | 32.276166 ms |
| Two-method evaluation wall time, 128 records | 8.171083 ms |
| Prediction latency, 512 calls | p50 0.004375 ms; p95 0.005625 ms; p99 0.006291 ms; max 0.162333 ms |
| Whole benchmark process peak RSS | 43,663,360 bytes, about 41.6 MiB |

The process included the Python test-fixture/pytest import used to construct
records. These observed workstation timings and memory usage are bounded
plumbing measurements; deployment contention, thermal behavior, coprocessor
latency and physical behavior remain unmeasured.

## Requirements for a useful learned successor

Collect controller-owned logs without advisory predictions influencing commands
first. Bind candidate/task context to accepted commands, actual execution
deviations, causal ego state, track snapshots, mechanism positions/modes,
fresh possession/readiness feedback, cancellation/replacement and outcome
labels. Include failed acquisitions, interlock stops, changing targets,
observation dropout, localization resets, payload differences and relevant
field/mechanism configurations. Freeze success/utility/risk definitions and
label sources before scoring. Additional mechanism state belongs in an
explicitly versioned owner contract, not reconstructed zero-filled features.

A practical successor can retain the short-horizon ego dynamics GRU, add a
separate structured task/context encoder, and train distinct completion,
possession/readiness and utility/risk heads with observed labels. Variable
object sets need identity-aware masked encoding; model predictions never update
authoritative observed tracks. Compare independent ensemble members versus the
current shared bases deliberately, since shared heads can understate common
model error. Fit normalization on train only; evaluate action/context ablations
and horizon-specific error on disjoint sessions/configurations. Keep the
physical transition and task-value losses/labels separately reported.

This new task-outcome record contains a cutoff snapshot, not the complete fixed
GRU history. It cannot populate a learned RWM input by itself. Training a
successor that uses the recurrent dynamics requires joining the record to
validated ego-state history and the complete H-1 historical accepted-command
intervals from the existing strict local data/window pipeline, with matching
request/session/source/reset identities and synchronization. Missing history,
future-smoothed state, gaps or changed identities must reject that joined
example. No missing history is padded with zero, copied from future labels or
silently fabricated here.

Probability reliability, uncertainty coverage, OOD abstention, delayed feedback
and unsafe false negatives need held-out calibration and frozen acceptance
limits. Task utility heads trained on executed logs estimate outcomes under the
logged selection policy; their ranking of unexecuted goals remains affected by
selection bias. Demonstrating ranking improvement needs authorized controlled
comparison or validated paired simulation, then separate robot-owned shadow
and guarded hardware evidence. None is supplied here.

No authorized real task-outcome logs, matched FRC outcome checkpoint, calibrated
task probabilities/covariance, independent counterfactual evidence, or physical
pickup/scoring validation were supplied. The live learned path therefore
remains unavailable. Current lightweight tooling stays on the workstation;
future expensive learned inference belongs off the roboRIO periodic loop on a
qualified coprocessor. Robot-owned freshness, interlocks, path checks and
reactive fallback remain required when that service is delayed or unavailable.
