# Exact World-State boundary

Contract owner: FRC-World-State. Semantic pin:
`1f8b9aace3fd320e3545744e3125aa348089bc99`. The owner's later stack pin
`209743312e33ec926d2f20e995eaca8daa2f8742` leaves the consumed API/schema/vector
content unchanged. The tested semantic pin is retained.

`contracts/world-state-v1/manifest.json` records exact SHA-256 for the exported
schemas, golden vectors and contract document. Package resources contain the
same bytes. Request schema hash is `4fe241f505bc6055d9526bfb7dda996a7838cfcbdc12f5c636989cc9c1d5f658`;
reply schema is `5d05c7fef9f08ed1d4196c64455c623b6cd1b489852fa609c65bccab03e6cc14`.
Golden request is `d232d9d8b5aac65f8ac2db191bfeb03d0d3c0e23301333fb3039142f735e4b4b`;
golden reply is `7e2a3252ca2792d80fdaa700e778de5c3d8a547328cb171af575df895e61bc75`.

`bindings.py` validates actual Draft 2020-12 schemas and additional semantics:
finite JSON (including overflow), duplicate keys, fixed integer microsecond
clock/grid, synchronized causal history, PSD covariance, accepted action times,
unique identities, configured physical domains, source/epoch/frame/cutoff and
expiry. There is no default physical gate policy. Production limits need robot
measurements; fixture policies are explicitly synthetic.

A reply matches the original request snapshot and cutoff. A newer current
snapshot can pass only the same epoch/field/obstacle map, monotonic fresh valid
ego and retained requested track identities. Remaining samples use absolute
`history_cutoff_us + offset_us`; they are never rebased to arrival/current time.
The exact Java owner retains authority for applying forecasts.

## Model input and output coverage

| Boundary | Supported contents |
|---|---|
| Wire history | Ego field x/y/heading, field velocity converted explicitly to robot-relative vx/vy, omega; history validity and accepted robot-relative commands |
| Learned ego input | Ordered `(x_m, y_m, heading_rad, vx_robot_mps, vy_robot_mps, omega_radps)` |
| Learned action | Ordered `(vx_robot_mps, vy_robot_mps, omega_radps)`, physical accepted interval commands in logs and expected candidate interval commands at inference |
| Candidate grid | `0, step_us, ..., horizon_us`; final endpoint has no following model interval and is not an additional transition |
| Learned outputs available in wire v1 | Field position/velocity, validity and separately calibrated PSD positional covariance |
| Local synthetic mechanism exercise | Eight state/four action layout, independent private envelope, explicitly pending production contract match |
| Missing v1 prerequisites | Mechanism state, heading/mechanism forecast outputs, pickup labels, robot/session/day identity, ego capture provenance, complete command-stream assurance, saturation/execution deviations |

`candidate_body_actions3` and `prepare_wire_inputs` reject an eight-state/four-action
model, configured mechanisms that would be discarded, unknown history, missing
accepted commands, mismatched fixed timestep, command changes inside a model
interval, and short/gapped histories. They never pad missing values with zeros
or use future measured velocity/state as an action. A planner/controller-owned
adapter must produce the same timed candidate command sequence used in training.
No A* waypoint-to-command adapter is invented here.

The default published architecture configuration remains two GRU256 layers,
128-unit heads and five members. A matched wire model changes task dimensions
only to six/three; history/timestep/horizon remain explicit frozen configuration.
The golden vector has mechanisms and a short history, so it tests serialization
and gating, not adequate learned-model input.

## Log conversion

`tools/convert_world_state_log.py LOG --metadata METADATA.json --output-directory NEW_DIR`
requires metadata keyed by request ID. Each entry supplies actual `robot_id`,
`session_id`, ISO `day`, `calibration_id`, `source_id`, `label_provenance`,
`ego_capture_us_by_snapshot`, integer `time_sync_limit_us`, `input_causal: true`
and `complete_accepted_command_history: true`. These are source attestations,
not values this converter infers. Fused estimates stay fused estimates.

Only history `ACCEPTED` commands enter executed-action records. Candidate
commands remain hypothetical and never become outcome labels. Accepted commands
are held until the next known accepted command only after complete-stream
assurance; the terminal command with unknown interval is excluded. Clipping,
saturation and execution differences cannot be reconstructed from v1.
Overlapping requests deduplicate physical snapshots/commands while merging
request provenance; conflicting facts fail. Original files remain unchanged.

Exact validated microseconds multiply by 1000 into explicitly named local
nanoseconds; the wire domain/mapping is retained. 2027 NT nanosecond metadata is
not a reason to change JSON `_us`.

## Replay/shadow abstention

`tools/replay_wire.py REQUEST --maximum-sync-error-us LIMIT --output NEW_REPLY`
validates at the recorded issue time and produces a schema-valid masked reply.
The wire has no `status` field. Abstention enumerates every candidate/target pair
and every grid point, uses canonical masked zeros/`confidence: null`, and names
`uncalibrated-abstention-v1`. No missing forecast becomes measured state.

`FrozenWireShadow` loads checksum-validated CPU weights and train-only
normalization from a local package. With matched ego6/body3 layout, complete
history prerequisites and mapped robot time, it runs native independent
candidate rollouts and retains raw, uncalibrated diagnostics. Its wire replies
**always abstain in this version**, including overload or missing inputs.
There is no physical-covariance adapter or live promotion bypass. External tracks
and every masked candidate/target continue using World-State's analytical baseline.

`service.py` separately provides a typed private replay harness for the pending
synthetic ego/mechanism envelope. Its internal status/epoch fields are not a
second World-State wire contract. It demonstrates bounded work, cancellation,
crash/timeout/epoch handling and analytical fallback without a command interface.

`tools/verify_cross_language.py WORLD_STATE_CHECKOUT --source-ref COMMIT_SHA --output-directory NEW_DIR`
reads the pinned Java source, compiles into this task, verifies Java's 109 gate
assertions, compares emitted request/reply bytes to the copied golden vectors,
and consumes them in Python. A matching Java/Python abstention reply is checked
for semantic equality; Java's gate reports `UNCALIBRATED` with baseline retained.
This proves Java emission/Python consumption and shared reply semantics. It does
not claim a Java JSON parser that the owner has not implemented.

The optional Java cross-check uses the separately maintained public source at
[Antigro09/FRC-World-State](https://github.com/Antigro09/FRC-World-State).
Pass an exact 40-character public commit with `--source-ref`; source files must
match `java-source-manifest.json`. The historical semantic commit above records
the original verification and is not assumed available in a sanitized public
history. An explicit `--working-tree` local checkout override is also supported
and verifies those same source hashes. No private absolute path is a default.
The ordinary Python setup, test suite and package build are self-contained and
require no other component checkout or published Maven artifact.

Verified reachable public source commit:
[`0e5b6c3f85d47205cde8b0c80e13fa79d35e95ab`](https://github.com/Antigro09/FRC-World-State/tree/0e5b6c3f85d47205cde8b0c80e13fa79d35e95ab).
A fresh public checkout matched every consumed Java/schema/fixture hash and
passed the cross-language check. The historical semantic pin and exact v1 bytes
are retained; this adds no model runtime dependency and requires no circular
repinning of other repositories.

```sh
git clone https://github.com/Antigro09/FRC-World-State.git artifacts/world-state
git -C artifacts/world-state checkout --detach 0e5b6c3f85d47205cde8b0c80e13fa79d35e95ab
.venv/bin/python tools/verify_cross_language.py artifacts/world-state --source-ref 0e5b6c3f85d47205cde8b0c80e13fa79d35e95ab --output-directory artifacts/public-cross-language
```

Use new output directories. The explicit hash-verified working-tree override
remains available; ordinary Python tests/build remain self-contained.
