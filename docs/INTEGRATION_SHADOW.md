# Independent vision and bounded shadow process

The approved software integration uses two independent Custom Vision consumers:
one feeds roboRIO localization directly; the other feeds the model's own bounded
inbox. The model never relays or acknowledges the robot's localization stream.
roboRIO owns the pose estimator and actuators. This implementation covers only
simulation, replay and shadow review; it enables no hardware deployment or motion.

`VisionSource(root, pipeline, kind)` makes every coherent `/result` topic explicit.
`VisionInbox` validates the pinned `custom-vision-schema2-2026.1` schemas, original
publication metadata, source/boot/packet/frame ordering and compact target/capture
references. Schema files are packaged and SHA-256 checked at startup. The copies
come from the existing Custom-Vision `protocol` tree; no producer schema changed.
Receipt monotonic nanoseconds, producer capture microseconds and original NT
server publication microseconds remain distinct. Empty/error/disconnected packets,
source queue loss, possible native polling loss, receipt expiry and reconnects
invalidate that source. Two advancing publications adopt a boot/connection/loss
session; a tombstoned frame cannot revive. Retired boot memory is capped and fails
closed at its ceiling. Revision changes require a new producer boot.

The bounded observation store is shadow evidence. `trace_for_request` records
provenance and requires an explicit `map_capture(source, nt_server_us)` callback
before comparing to a robot cutoff. Missing mapping stays ineligible. Callbacks
run outside the inbox lock; source/epoch/expiry is rechecked before returning.
No arrival-time substitution, field projection, ego fusion or track ownership is
performed. Raw vision cannot replace robot-owned measured facts or become a
second pose estimator. Object coordinates and covariance stay in the declared
capture frame; no model physical covariance is invented.

`PyntcoreAdapter` owns a newly created instance, connection listener and string
subscriptions. It publishes nothing and uses no HAL/camera APIs. Its only accepted
Python timestamp profile is `pyntcore==2024.3.2.1`; other versions fail before client
creation. Missing server timestamp metadata is rejected, never replaced with local
NT time. `VisionSubscription` polls on its own thread, fences full native batches
and reconnects during a drain, and performs no inference, logging or transport
roundtrip. It is separate from the robot client's storage and lifetime. The
optional native test dependency lock is `requirements-vision-nt4-test.lock`.
The old Python client profile is deliberately independent of the Java season
stack; this does not qualify any Jetson image or hardware compatibility.

`ShadowRuntime(policy, authority, now_us, inbox, subscription=None)` starts an
optional independent subscription and serves exact `frc-decision-support/1`
**baseline-only** replies. The callback `now_us` must be an explicitly mapped
robot clock or a declared replay clock. Robot-owned snapshot, measured context,
occupancy and task/path facts remain unchanged. Direct vision is recorded in
`latest_trace()` separately from the decision wire schema. `service_epoch` is a
fresh boot fence. Saved requests and every reply need exact original/current
identity, task/goal, scoring and expiry checks in the robot consumer.

An explicitly supplied learned provider can be supervised separately with
`ProcessPredictor(provider, timeout_s=..., startup_timeout_s=..., max_bytes=...)`.
One spawned process handles one request. Bounded JSON frames on a nonblocking
socket pair avoid blocking `multiprocessing.Pipe.send` or partial `recv` after a
child stalls. Serialization, IPC send/read and cancellation share the request
wall-clock deadline; child terminate/join/kill has a separate <=100ms cleanup
budget. Timeout/crash retires that child; `restart()` is explicit outside the
request deadline. Generation and framing checks reject stale/corrupt IPC.
No provider or trained weights are selected by default. `DecisionService` keeps
its baseline when optional evidence is missing, unqualified, timed out or rejected.
OS memory/process-tree limits, actual calibrated model evidence and device-load
qualification remain necessary before any later hardware use.

## Local process replay

Run only with explicit synthetic fixture inputs and new output paths:

```sh
.venv/bin/python tools/serve_shadow.py \
  --request contracts/decision-support-v1/request-v1.json \
  --policy configs/decision-synthetic.json \
  --descriptor /tmp/frc-shadow-boot-a.json \
  --trace /tmp/frc-shadow-boot-a.ndjson
```

The tool opens only an ephemeral `127.0.0.1` NDJSON socket. It creates no NT client
or camera/controller connection. The descriptor declares a fixed synthetic robot
clock (`issued_us` from the fixture), numeric host/port and fresh `service_epoch`.
Copy that epoch into the client's saved fixture request before serialization.
A restart uses a new descriptor path and epoch; old requests are rejected. The
fixed fixture clock must never be presented as a live robot clock or used for
real-data freshness decisions.

`transport.roundtrip` uses one monotonic exchange budget covering connect,
write and all reads, with no DNS resolution. Server reads and writes share one
exchange budget, so slow-drip peers cannot refresh the timeout. The server owns
one serialized daemon and a two-connection listen backlog. Shutdown closes the
listener and active socket and joins for at most 200ms. A trusted uncooperative
handler can remain in that one daemon; it cannot be stopped by a Python thread.
Run untrusted or native inference in the process supervisor, not in the handler.

## Evidence

Synthetic tests cover coherent object references, frame tombstones, expiry,
boot/revision retirement, reconnect adoption, queue/native saturation and callback
fence races. Actual spawned-process tests cover IPC overload/cancellation,
provider crash, oversized input/output, `SIGSTOP` during a large send, kill and
explicit restart. An actual localhost shadow process is stalled, killed and
restarted while a separate vision consumer processes 40 observations; old service
epochs are rejected. The optional native NT4 test uses two distinct clients against
an isolated localhost server and verifies original server timestamps and model
client disconnect/reconnect while the other consumer continues.

These results demonstrate software isolation and fallback. They are not robot
physical calibration, learned accuracy, live NT compatibility with an existing
Jetson installation, or permission to deploy or actuate.
