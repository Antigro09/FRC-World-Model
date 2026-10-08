#!/usr/bin/env python3
"""Read pinned Java source; compile only into this task; verify exported vectors."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frc_world_model.bindings import CONTRACT,load_document,make_masked_reply,parse_request

PIN="1f8b9aace3fd320e3545744e3125aa348089bc99"
JAVA_FILES=["src/main/java/org/frcworldstate/core/"+n+".java" for n in ("Geometry","World","Predictor","PredictionGate")]+["src/test/java/org/frcworldstate/core/PredictionContractTest.java"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("world_state_checkout",type=Path)
    p.add_argument("--output-directory",type=Path,required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-ref",help="Exact 40-character published or local commit SHA")
    source.add_argument("--working-tree",action="store_true",help="Explicit local checkout override, verified by source hashes")
    args=p.parse_args()
    manifest=json.loads((CONTRACT/"manifest.json").read_text())
    for name,sha in manifest["files"].items():
        if hashlib.sha256((CONTRACT/name).read_bytes()).hexdigest()!=sha:raise ValueError("consumed contract checksum mismatch")
    # Compare latest owner's stable contract content, retaining semantic pin.
    for name in ("prediction-request-v1.schema.json","prediction-reply-v1.schema.json","request-v1.json","reply-v1.json"):
        folder="schemas" if ".schema." in name else "fixtures/prediction"
        if hashlib.sha256((args.world_state_checkout/folder/name).read_bytes()).hexdigest()!=manifest["files"][name]:
            raise ValueError("current owner contract changed")
    args.output_directory.mkdir(parents=True,exist_ok=False)
    src=args.output_directory/"java-source";src.mkdir()
    sources=[]
    expected=json.loads((Path(__file__).resolve().parents[1]/"contracts/world-state-v1/java-source-manifest.json").read_text())["source_sha256"]
    if args.source_ref and not re.fullmatch(r"[0-9a-fA-F]{40}",args.source_ref):
        raise ValueError("source-ref must be an exact immutable commit SHA")
    for file in JAVA_FILES:
        value=(args.world_state_checkout/file).read_bytes() if args.working_tree else subprocess.check_output(["git","-C",str(args.world_state_checkout),"show",args.source_ref+":"+file])
        if hashlib.sha256(value).hexdigest()!=expected[file]:
            raise ValueError("World-State source differs from consumed semantic contract: "+file)
        destination=src/Path(file).name;destination.write_bytes(value);sources.append(str(destination))
    classes=args.output_directory/"classes";classes.mkdir()
    subprocess.run(["javac","--release","17","-d",str(classes),*sources,str(Path(__file__).resolve().parents[1]/"tests/java/AbstentionCrossCheck.java")],check=True)
    emitted=args.output_directory/"java-vectors"
    subprocess.run(["java","-cp",str(classes),"org.frcworldstate.core.PredictionContractTest","--write-fixtures",str(emitted)],check=True)
    subprocess.run(["java","-cp",str(classes),"org.frcworldstate.core.PredictionContractTest",str(CONTRACT)],check=True)
    for kind in ("request","reply"):
        if (emitted/f"{kind}-v1.json").read_bytes()!=(CONTRACT/f"{kind}-v1.json").read_bytes():
            raise ValueError("Java emission differs from consumed golden fixture")
        load_document((emitted/f"{kind}-v1.json").read_text(),kind)
    request=json.loads((CONTRACT/"request-v1.json").read_text())
    parse_request(json.dumps(request),now_us=request["issued_us"],max_sync_error_us=1000)
    abstention=args.output_directory/"java-abstention.json"
    subprocess.run(["java","-cp",str(classes),"AbstentionCrossCheck",str(abstention)],check=True)
    python_reply=make_masked_reply(request,request["issued_us"])
    assert load_document(abstention.read_text(),"reply")==python_reply
    (args.output_directory/"python-abstention.json").write_text(json.dumps(python_reply,allow_nan=False)+"\n")
    print(json.dumps({"semantic_pin":PIN,"source_ref":args.source_ref or "explicit byte-verified local working tree","golden_java_emission_python_consumption":"passed","abstention_semantic_parity":"passed","java_gate_baseline":"UNCALIBRATED","java_json_parser":"not implemented/claimed","output":str(args.output_directory.resolve())}))


if __name__=="__main__":main()
