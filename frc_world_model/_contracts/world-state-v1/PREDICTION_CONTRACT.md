# Prediction contract `frc-prediction/1`

The optional predictor depends on the pure Java core. It cannot edit authoritative robot facts, declare task success, command hardware, or change tracks. A reply contains **LEARNED_FORECAST** values. Observations, tracked estimates, deterministic extrapolations, and forecasts remain distinct.

Wire documents are `schemas/prediction-request-v1.schema.json` and `schemas/prediction-reply-v1.schema.json`. Golden vectors are `fixtures/prediction/request-v1.json` and `reply-v1.json`. Contract ownership is FRC-World-State. Candidate/model changes that require a different layout require a schema version change.

## Frames, time, identity

Positions and forecast velocities are SI in the fixed `BLUE_FIELD` frame: the configured blue origin, +x down field, +y left, CCW radians. An alliance transform happens once at the robot boundary. `field` includes season, map ID and geometry revision; the season does not select a controller toolchain.

All JSON `*_us` values use integer microseconds in `ROBOT_MONOTONIC_US`. They are never NT publication timestamps or nanosecond API values. Normalized samples include a synchronization source, mapping revision, validity and maximum timing error. The gate requires valid synchronization inside its configured error bound. `history_cutoff_us` bounds the data used to compute a reply, independently of request issue, reply generation, and expiry. Source context carries robot boot, localization and configuration revisions.

Wire identities contain at most 256 Unicode code points. All wire integers are nonnegative and at most `9007199254740991`, preserving exact integer JSON interchange. These bounds are enforced when Java wire records are created and by the exported schemas. Invalid forecast values remain representable until the consumption gate rejects them.

Every request/reply identifies schema, model, request, snapshot, epoch and field, and repeats source context, cutoff, horizon, step and time/frame domains. The reply matches the original request snapshot exactly. A newer current snapshot is allowed only with the same epoch, field and obstacle-map version, monotonic/fresh ego time, valid localization, and retained requested track IDs, within configured history age, forecast horizon and expiry. Hard resets, source boot changes, changed geometry/configuration and task replacement invalidate pending work. Histories are immutable and bounded to 128 snapshots, 256 tracks per snapshot and 256 provenance stamps per track; requests allow 32 targets, 16 candidates, and 256 samples per candidate/forecast. These are computational bounds, not measured hardware safety limits.

## History and actions

Targets have an explicit `EGO` or `OBJECT_TRACK` kind, stable string target ID, and a track ID only for object targets. History entries carry an immutable snapshot and a validity mask. Invalid history entries are excluded from learned inputs. They are not evidence that space is empty. Track provenance retains capture/publication time, source/boot, correlation group, calibration, mount revision and target-height assumption. A model must not treat correlated camera estimates as independent measurements.

An optional history `accepted_command` records robot-owned, physically timestamped commands with `issued_us`, `accepted_us`, `source_id`, `command_id`, `ROBOT_RELATIVE` frame (+x forward, +y left, CCW positive), raw SI `vx_mps`, `vy_mps`, `omega_radps` and configured mechanism values with explicit units. Only the robot execution boundary may label commands accepted. An accepted command is not measured motion or proof of task completion.

Request action candidates have `CANDIDATE` authority and `BODY_TWIST_OPEN_LOOP_V1` semantics: raw robot-relative body twist sampled on the request's fixed time grid, held until the next sample. Mechanism channels are robot configuration with explicit units (`m`, `m/s`, `rad`, `rad/s`, `ratio`, `bool`). Candidates describe hypothetical intended actions; they have not been accepted by the robot. They do not prescribe vendor motor outputs. The predictor evaluates each candidate independently from the same cutoff. V1 has no probabilistic policy or rollout feedback semantics.

## Replies and gating

There is exactly one forecast for every requested candidate/target pair and exactly one sample at offsets `0, step_us, ..., horizon_us`. Each sample's absolute robot time is `history_cutoff_us + offset_us`; consumers skip elapsed samples and never rebase offsets onto a newer snapshot or reply time. A forecast sample has a validity mask, field position/velocity and a 2D positional covariance in m². Masked-out numeric values are canonical zero placeholders; masked-out samples cannot be followed or interpreted as free space. The gate requires at least one remaining valid sample; each missing candidate/target future uses its own baseline. Heading and mechanism-state prediction are outside v1; a new schema is required to add them.

Finite, symmetric positive-semidefinite covariance is required for every valid sample, with minimum and maximum eigenvalues inside configured variance bounds. A reply must name a separately approved uncertainty calibration ID bound to its model. Confidence alone cannot substitute for calibrated uncertainty. The gate pins allowed field and robot configuration identities and bounds forecast displacement from input anchors and between valid samples. Approval is configuration evidence, not a claim this repository has calibrated a learned model.

The gate rejects absent, unsupported, stale/future, mismatched, out-of-domain, nonfinite, dimensionally inconsistent, unsynchronized, invalid-mask or uncalibrated replies. Physical/input/output limits are explicit gate configuration and must later come from robot measurements. Missing/rejected predictions leave deterministic selection, tracking and planning behavior available. Accepted forecasts are advisory; collision supervision still uses its independent conservative occupancy model.

Model/network/file I/O and training stay outside the periodic path. An injected worker must bound execution, use latest-request-wins replacement, honor cancellation, expose overload, and discard old identities. The predictor SPI does not imply a controller/model runtime implementation.

## Reproducible verification

`PredictionContractTest` generates the golden JSON using the Java records and validates rejection cases. `python3 scripts/verify_prediction_fixtures.py` independently parses and checks both wire vectors without third-party dependencies. Java compares its emitted vectors byte-for-byte with the checked-in vectors; Python verifies JSON round-trip equality and the contract dimensions/units/masks. This proves Java emission/Python consumption, not a Java wire parser that has not been implemented.
