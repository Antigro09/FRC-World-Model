#!/usr/bin/env python3
"""Small native CPU RWM latency check, including queue and serialization.

This Mac synthetic test cannot qualify Jetson latency, power, thermal behavior,
or simultaneous perception load. No model weights are downloaded at runtime.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from frc_world_model.rwm import FrozenRWM, ModelConfig, make_model
from frc_world_model.frozen import load_frozen_model


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered)-1,int((len(ordered)-1)*fraction+.5))]


def benchmark(output, *, samples=30, warmup=3, candidates=2, horizon=8, package=None):
    if not 1 <= samples <= 100 or not 0 <= warmup <= 20 or not 1 <= candidates <= 8:
        raise ValueError("small CPU scope limits exceeded")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if package:
        _, predictor = load_frozen_model(package)
        config = predictor.config
        weights = "checksum-validated frozen package"
    else:
        config = ModelConfig()
        predictor = FrozenRWM(make_model(config),config)
        weights = "random synthetic initialization; performance plumbing only"
    if not 1 <= horizon <= config.forecast_horizon:
        raise ValueError("unsupported horizon")
    history = torch.zeros(1,config.history_horizon,config.state_dim)
    past_actions = torch.zeros(1,config.history_horizon,config.action_dim)
    commands = torch.zeros(1,horizon,config.action_dim)
    def job(queued_ns):
        worker_started = time.perf_counter_ns()
        forecasts = []
        for candidate in range(candidates):
            result = predictor.rollout(history,past_actions,commands + candidate*.01)
            forecasts.append({"candidate_id":str(candidate),"states":result.member_states.tolist(),
                              "normalized_stds":result.member_stds.tolist(),"uncertainty":result.uncertainty_semantics})
        inferred = time.perf_counter_ns()
        payload = json.dumps(forecasts,allow_nan=False,separators=(",",":"))
        decoded = json.loads(payload)
        ended = time.perf_counter_ns()
        return {"queue_ms":(worker_started-queued_ns)/1e6,"native_inference_ms":(inferred-worker_started)/1e6,
                "serialization_ms":(ended-inferred)/1e6,"end_to_end_ms":(ended-queued_ns)/1e6,"bytes":len(payload.encode())}
    timings = []
    with ThreadPoolExecutor(max_workers=1) as worker:
        for index in range(warmup+samples):
            queued = time.perf_counter_ns()
            result = worker.submit(job,queued).result()
            if index >= warmup:
                timings.append(result)
    budget_ms = config.timestep_us/1000
    metrics = {}
    for key in ("queue_ms","native_inference_ms","serialization_ms","end_to_end_ms"):
        values = [r[key] for r in timings]
        metrics[key] = {"p50":quantile(values,.5),"p95":quantile(values,.95),"p99":quantile(values,.99),"max":max(values)}
    report = {"host":{"platform":platform.platform(),"architecture":platform.machine(),"python":platform.python_version(),"torch":torch.__version__},
              "device":"cpu","threads":1,"weights":weights,"warmup":warmup,"samples":samples,
              "candidates":candidates,"horizon":horizon,"resolved_config":config.resolved(),"timings":metrics,
              "budget_ms":budget_ms,"deadline_misses":sum(r["end_to_end_ms"]>budget_ms for r in timings),
              "max_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=="Darwin" else 1024),
              "serialized_bytes":timings[0]["bytes"],"power":"unknown","thermal":"unknown",
              "simultaneous_vision_load":"unrun","jetson_qualification":"unrun","live_advisory_promotion":False}
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--package",type=Path)
    parser.add_argument("--samples",type=int,default=30)
    parser.add_argument("--candidates",type=int,default=2)
    args = parser.parse_args()
    result = benchmark(args.output,samples=args.samples,candidates=args.candidates,package=args.package)
    print(json.dumps({"output":str(args.output.resolve()),"timings":result["timings"],"deadline_misses":result["deadline_misses"]}))
