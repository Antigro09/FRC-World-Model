"""Independent coherent vision input plus baseline-only shadow decision service."""
import copy
import threading

from .bindings import ContractError
from .decision_bindings import canonical_json, load_decision
from .decision_service import DecisionService


class ShadowRuntime:
    """Robot-owned snapshots remain authoritative; direct vision is trace evidence.

    Vision is subscribed through a distinct client with its own bounded inbox.
    Serving, logging, and optional model IPC have no ownership of roboRIO's
    independent subscription or pose estimator. This baseline-only integration
    records direct observations separately; it never fuses them into robot ego
    or fabricates tracks, prerequisites, calibrated covariance or learned data.
    now_us must be an explicit mapped robot clock (or an explicit replay clock).
    """
    def __init__(self, policy, *, authority, now_us, inbox, subscription=None):
        if subscription is not None and subscription.inbox is not inbox:
            raise ValueError("subscription must own this independent model inbox")
        self.inbox, self.subscription = inbox, subscription
        self.service = DecisionService(policy, authority=authority, now_us=now_us)
        self._trace_lock, self._trace, self._closed = threading.Lock(), None, False
        if subscription is not None:
            try:
                subscription.start()
            except Exception:
                self.service.close()
                subscription.close()
                raise

    @property
    def service_epoch(self):
        return self.service.service_epoch

    def evaluate(self, payload):
        if self._closed:
            raise ContractError("shadow runtime stopped")
        request = load_decision(payload, max_bytes=self.service.policy.max_bytes)
        reply = self.service.evaluate(payload)
        trace = self.inbox.trace_for_request(request)
        trace.update(service_epoch=self.service_epoch, decision_generation=request["decision_generation"],
                     decision_status=reply["status"], vision_is_shadow_evidence=True)
        with self._trace_lock:
            self._trace = trace
        return reply

    def __call__(self, payload):
        return canonical_json(self.evaluate(payload))

    def latest_trace(self):
        with self._trace_lock:
            return copy.deepcopy(self._trace)

    def diagnostics(self):
        return {"service_epoch": self.service_epoch, "baseline_only": True,
                "influences_commands": False, "inbox": self.inbox.diagnostics(),
                "subscription": self.subscription.diagnostics() if self.subscription else None}

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.service.close()
        if self.subscription is not None:
            self.subscription.close()
        else:
            self.inbox.close()
