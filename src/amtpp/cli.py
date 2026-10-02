"""Run a registered AMTPP configuration with explicit local input/output paths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the saved best checkpoint on test events")
    parser.add_argument("--topology", type=Path, help="Optional station topology NPZ")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if set(config) != {"protocol", "arguments"} or config["protocol"] not in {"strict", "legacy"}:
        parser.error("Configuration must contain protocol (strict or legacy) and arguments")
    if not isinstance(config["arguments"], dict):
        parser.error("Configuration arguments must be an object")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    strict = config["protocol"] == "strict"
    module = "amtpp.train" if strict else "amtpp.legacy.evaluate" if args.evaluate else "amtpp.legacy.train"
    command = [sys.executable, "-m", module, "--data", str(args.data.resolve()), "--device", args.device, "--seed", str(args.seed)]
    if strict:
        command += ["--run-id", output.name, "--checkpoint", str(output / "best.pt"), "--result", str(output / ("test.json" if args.evaluate else "selection.json"))]
        if args.evaluate:
            command += ["--evaluate-only", "--final-test", "--event-artifact", str(output / "events.npz")]
        if args.topology:
            command += ["--topology", str(args.topology.resolve()), "--topology-mode", "graph"]
    else:
        if args.topology:
            parser.error("The network adapter is supported by the strict entry point")
        command += ["--checkpoint", str(output / "best.pt")]
        if args.evaluate:
            command += ["--result", str(output / "test.json")]
    reserved = {"data", "device", "seed", "run-id", "checkpoint", "result", "event-artifact", "evaluate-only", "final-test", "topology", "topology-mode"}
    training_only = {"epochs", "patience", "lr"}
    for key, value in config["arguments"].items():
        if key in reserved or not isinstance(key, str) or key.startswith("-"):
            parser.error(f"Invalid configuration argument: {key}")
        if not strict and args.evaluate and key in training_only:
            continue
        if value is True:
            command.append("--" + key)
        elif value is not False and value is not None:
            command += ["--" + key, str(value)]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
