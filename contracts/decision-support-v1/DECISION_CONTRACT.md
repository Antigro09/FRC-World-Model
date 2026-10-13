# Advisory decision support `frc-decision-support/1`

This companion contract proposes which supplied high-level task to consider next.
It separates measured World-State snapshots, hypothetical body-twist actions,
predicted outcomes, task ranking, path planning and guarded execution. A reply
does not modify tracks, estimate robot pose, command mechanisms or complete a
task. Every reply fixes `measured: false`, `influences_commands: false` and
`requires_planner_validation: true`. Supported modes are `replay` and `shadow`.

The authoritative predictor contract remains World-State `frc-prediction/1`.
`prediction_request` embeds its exact fields and copied local definitions. The
schemas contain no remote references and require no network resolution. The
generation source is `frc_world_model/decision_schema.py`; it verifies the pinned
request-schema SHA256 before copying definitions. The enclosing profile narrows
history to 32 samples and candidates to eight without changing the original
prediction schemas or their golden vectors.

## Request and identities

`decision-request-v1.schema.json` requires every request field. Unknown fields
are rejected at all defined object boundaries. The request carries:

- `service_epoch`, `decision_generation`, `task_revision` and `policy_id`, binding
  one service boot, supersession generation, candidate/task configuration and
  configured ranking policy. Service restart, cancellation, task replacement and
  authoritative context changes invalidate outstanding advice.
- `prediction_request`, retaining schema/model/request/snapshot/epoch, blue field
  identity, robot boot/localization/configuration revisions, robot-monotonic
  microsecond timestamps, valid clock mapping, immutable history, persistent
  targets and independent hypothetical action candidates.
- `action_sequence_hash`, lowercase SHA256 of canonical JSON for
  `prediction_request.candidates`. Canonical JSON uses UTF-8,
  `json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)`
  with Python's default `ensure_ascii=True`. Candidate order is significant.
- `context`, with its revision, measurement time and source ID; measured
  possession `EMPTY`/`HELD` or unavailable `UNKNOWN`; measured readiness boolean
  or unavailable `null`; enable, command authority and cancellation facts.
- `current_task_id`, nullable; independently supplied `obstacles`; and one to
  eight supplied `tasks`.

`EMPTY` requires actual measured absence. Missing robot possession sensing maps
to `UNKNOWN`; missing readiness sensing maps to `null`. A command, motor current,
path endpoint, elapsed predicted duration or disappearing visual target cannot
create these measurements. A source ID and timestamp describe supplied evidence;
the service does not establish sensor validity or fabricate a sensor binding.

Context and obstacle measurements must be causal at or before the embedded
history cutoff and fresh under the explicit service policy. A revision change
must accompany changed possession, readiness or task context. All JSON integers
are nonnegative and at most `9007199254740991`. Identities contain at most 256
Unicode code points. Parsing additionally rejects duplicate keys, nonfinite
numbers and oversized documents. Resource limits and schema limits are upper
bounds for engineering work, not physical safety qualification.

`obstacles` includes the World-State map version, measurement time, source ID,
`AVAILABLE`/`UNAVAILABLE` status and at most 128 field-frame circles. Each circle
has an ID, center, velocity and nonnegative radius. These are explicit advisory
occupancy inputs; the decision service does not infer clear space from absent
tracks or an empty camera frame. Geometry/uncertainty/age coverage must come from
the authoritative producer. Advisory moving-circle screening does not replace
World-State's conservative planner envelopes or collision supervisor.

## Tasks, candidate actions and planner evidence

Each task binds a `task_id`, one embedded `candidate_id`, `kind`, nullable `goal`,
up to 32 unique persistent `target_track_ids`, finite configured `base_utility`
and `path_evidence`. Supported kinds are `PICKUP`, `SCORE`, `TRANSIT` and `HOLD`.
Task utility is a policy value, not a learned success probability or official
game-scoring assertion. Robot code owns rule/mode/strategy eligibility.

A goal carries field pose and nonnegative position, heading and velocity
tolerances. Blue field axes and SI units are inherited from World-State. An
alliance transform occurs once at the robot boundary. A* does not choose these
tasks. World-State already supplies persistent-object/cluster selection and
pickup lifecycle contracts that future robot wiring should reuse.

`path_evidence.status` is `UNASSESSED`, `REACHABLE` or `BLOCKED`; optional route
distance, planner request ID and expiry retain the distinction between a
straight-line estimate and planner evidence. A caller must supply truthful,
current evidence. The scalar evidence object carries no follower-ready path and
does not reproduce full planner validation. `REACHABLE` cannot authorize motion
or survive changed task, snapshot, field, source, obstacle map or expiry.

Candidate body twists remain hypothetical `CANDIDATE` authority with
`BODY_TWIST_OPEN_LOOP_V1` semantics. Samples describe held robot-relative SI
commands on the existing fixed grid. They are not vendor motor outputs or
measured motion. Each candidate starts from the same cutoff. The final sample
has no subsequent integration interval. A planner-to-timed-command adapter is
still required before using actual robot-planned actions as learned inputs;
waypoints or solver duration cannot substitute for that adapter.

## Reply and provenance

`decision-reply-v1.schema.json` requires the service/generation/task/policy fences,
mode, request SHA256, action hash, generation/expiry times and complete identity.
`request_sha256` uses the same canonical JSON function on the entire companion
request. The nested identity's `schema_version` stays `frc-prediction/1`; it also
includes original model/request/snapshot/epoch/field/source/frame/time domain,
history cutoff, obstacle-map version and context revision.

Status is `BASELINE`, `LEARNED_ADVISORY`, `REJECTED` or `CANCELLED`. The reply keeps
the selected task and the deterministic baseline's selected task separately,
along with `learned_status` and an optional top-level reason. A consumer checks
the saved original request and its current authoritative context before using
advice. Expired, reordered, cancelled or identity-mismatched replies are
unavailable advice; robot-owned reactive fallback remains responsible for what
happens next.

Each ranking entry contains its task/candidate/kind, eligibility and reasons,
nullable score and baseline score, forecast kind and fallback reason, nullable
goal, path-planning requirement and predicted outcome. Forecast kind is:

- `KINEMATIC_ASSUMPTION`: explicitly assumes candidate body-twist tracking and
  configured uncertainty/error growth. Every physical covariance is `null`.
- `LEARNED_FORECAST`: identifies learned evidence admitted through the original
  model/uncertainty gate. Approval remains bound to a specific model and
  calibration; normalized head spread or ensemble disagreement is insufficient.

`learned_prediction_reply` retains the complete unchanged `frc-prediction/1`
reply when learned evidence is used; `learned_domain_id` identifies its approved
domain. Both are nullable when learned evidence is unavailable. Every
`LEARNED_FORECAST` ranking requires a non-null original reply and domain ID.
Consumers revalidate the original reply against the saved embedded request,
approved model/calibration/domain policy and current authoritative snapshot;
reported outcome samples must agree with the original candidate's position,
velocity, validity and covariance. A companion claim of learned provenance or
copied covariance is insufficient by itself.

Outcome samples retain the original fixed-grid offsets and are anchored to
`history_cutoff_us + offset_us`; arrival time never rebases them. Samples carry
validity, field position/velocity and nullable physical positional covariance.
Masked values are canonical zero placeholders with `physical_covariance: null`.
Covariance for learned values still requires semantic PSD/domain/calibration
checks beyond the JSON schema.

Outcomes also report remaining goal distance, signed progress, nullable route
distance/travel-time proxy, explicit distance basis, nullable signed minimum
clearance and a configured uncertainty radius. `STRAIGHT_LINE_LOWER_BOUND` is
not geometric reachability; `PLANNER_EVIDENCE` remains a supplied scalar estimate.
Travel time is a proxy under configured assumptions, not a learned completion
time. `task_success_probability`, `task_completion_time_s` and
`mechanism_outcome` are always `null` in this version because the available
ego6/body3 model and data do not qualify those predictions.

## Unavailable service and integration placement

Inference, JSON processing, candidate evaluation and training belong off the
robot periodic path. The service uses bounded work and explicit cancellation;
timeout, overload, crash, missing learned evidence or rejected forecasts retain
deterministic advice where the authoritative context permits it. An absent or
unsafe authoritative context produces unavailable advice rather than invented
measurements or confidence. Host-monotonic processing budgets are separate from
robot-monotonic wire timestamps; no host clock substitutes for robot time.

Recommended placement is an optional offboard Python service. A robot-owned Java
adapter would retain the request/context, apply strict reply fences and feed
proposed task/goal choices to World-State and the existing A* planner. World-State
remains authoritative for observed objects, Drive remains the sole robot pose
estimator, and roboRIO owns actuator commands and per-cycle execution guards.
The robot must independently plan/validate/follow its path, align using fresh
measurements and verify physical success. This contract provides no live robot
transport, scheduler, follower, possession sensor or drivetrain qualification.

`request-v1.json` is a bounded synthetic three-task example with pickup, score
and hold candidates. It preserves original World-State observation provenance;
its context is fixture evidence, and its unassessed paths are explicit. Golden
reply evidence is produced by the actual runtime after schema validation.
Synthetic replay/schema tests demonstrate software behavior only. Learned
readiness additionally needs real causal accepted-action histories, source and
session/day separation, mechanism/possession/outcome labels, OOD evaluation,
held-out covariance calibration and robot/hardware validation.
