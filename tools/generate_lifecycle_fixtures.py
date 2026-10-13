#!/usr/bin/env python3
"""Emit lifecycle fixtures from pinned World-State Java into this repo or /tmp.

Reads a neighboring checkout using git show; writes no files there. Ordinary
Python regressions consume the checked-in output without a JDK or checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


PIN = "4590335421306da7964b5bd91dfa4da7922982ea"
ROOT = Path(__file__).resolve().parents[1]
JAVA_ROOT = "src/main/java/org/frcworldstate/core/"
COMPILED = ("Geometry", "World", "PoseHistory", "WorldEngine")
REFERENCED = (*COMPILED, "TargetSelector", "SafetySupervisor")


def generate(checkout, source_ref=PIN):
    if source_ref != PIN:
        raise ValueError("regeneration requires the documented immutable World-State source pin")
    hashes = {}
    with tempfile.TemporaryDirectory(prefix="world-engine-lifecycle-") as directory:
        output = Path(directory)
        sources = []
        for name in REFERENCED:
            path = JAVA_ROOT + name + ".java"
            content = subprocess.check_output(["git", "-C", str(checkout), "show", source_ref + ":" + path], timeout=10)
            hashes[path] = hashlib.sha256(content).hexdigest()
            if name in COMPILED:
                target = output / (name + ".java")
                target.write_bytes(content)
                sources.append(str(target))
        classes = output / "classes"
        classes.mkdir()
        harness = ROOT / "tests/java/WorldEngineLifecycleFixtures.java"
        subprocess.run(["javac", "--release", "17", "-d", str(classes), *sources, str(harness)], check=True, timeout=30)
        result = json.loads(subprocess.check_output(["java", "-cp", str(classes), "WorldEngineLifecycleFixtures"], timeout=10))
    return {"fixture_version": "world-engine-lifecycle/1", "synthetic_inputs": True,
        "source_repository": "FRC-World-State", "source_commit": PIN, "source_sha256": hashes,
        "harness_sha256": hashlib.sha256(harness.read_bytes()).hexdigest(),
        "semantics": {
            "lifecycle": "WorldEngine.publish: established tracks are COASTING whenever snapshot time exceeds last capture time",
            "target_selection": "TargetSelector.eligible: non-TENTATIVE, measurement age <= configured maximum, and bounded variance",
            "policy_limits": "Only synthetic configuration; engine expiry and decision freshness are separate limits",
        }, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world_state_checkout", type=Path)
    parser.add_argument("--source-ref", default=PIN)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = generate(args.world_state_checkout, args.source_ref)
    with args.output.open("x") as output:
        output.write(json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"source_commit": PIN, "scenarios": len(data["scenarios"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
