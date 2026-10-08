# Controller toolchains and timestamp boundary

This service has no WPILib/HAL/controller dependency. Its World-State contract
uses SI values, `BLUE_FIELD` and integer JSON `*_us` in
`ROBOT_MONOTONIC_US`. A game season identifies the map/rules, not the controller
toolchain. Offseason operation does not change that distinction.

| Requested target | Component status |
|---|---|
| WPILib 2026 + roboRIO | Primary immediate controller ecosystem; common Python service remains controller independent. Robot integration is on hold and has not been compiled/deployed here. |
| WPILib 2027 alpha-7 + Systemcore | Separate pinned experimental controller ecosystem. Official matrix associates Systemcore image >=14 with WPILib >=alpha-7. This service is protocol compatible at the explicit adapter boundary; no hardware qualification is claimed. |
| WPILib 2026 + Systemcore | **No official supported target found in the inspected compatibility matrix.** Do not invent HAL compatibility, backport a controller runtime, or label this target supported. |

The [official SystemcoreTesting matrix](https://github.com/wpilibsuite/SystemcoreTesting#compatibility)
was re-read from its actual README on 2026-10-08. It lists image <=9 with
2027 alpha-1/2, image 10–13 with 2027 alpha-5/6, and image >=14 with >=2027
alpha-7. There is no WPILib 2026 entry. This is a precise statement about that
official matrix, not a claim that an unsupported custom port is impossible.

2026 NT Java metadata uses the 2026 adapter's microsecond convention. The pinned
2027 alpha-7 Java API has nanosecond metadata. The Java vision/World-State tasks
own those version-specific adapters; this service must never guess a divisor
from the season, controller or magnitude. JSON `_us` remains microseconds in
both profiles. The offline converter multiplies exact validated integer
microseconds by 1000 for explicitly named internal nanoseconds and preserves
the original wire mapping revision/domain. It does not reinterpret publication
time as capture/state time.

No controller networking/topic, robot project, HAL rewrite, Jetson upgrade,
driver station operation or hardware actuation is included. Live advisory
promotion requires separate held-out, on-device and shadow evidence plus
explicit approval.
