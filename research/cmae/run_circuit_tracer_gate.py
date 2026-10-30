"""Record the 6B circuit-tracer gate without implying pretrained execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from research.cmae.circuit_tracer_adapter import (
    CircuitTracerRequest,
    run_circuit_tracer,
)
from research.cmae.run_counterfactual_assurance import _json_ready


def run_gate(cifar_path: Path, arithmetic_path: Path) -> dict[str, Any]:
    cifar = json.loads(cifar_path.read_text())
    arithmetic = json.loads(arithmetic_path.read_text())
    cifar_valid = all(run["counterfactual"]["primary_gate"]["pass"] for run in cifar["results"])
    arithmetic_valid = all(run["counterfactual"]["gate"]["pass"] for run in arithmetic["results"])
    upstream_valid = bool(cifar_valid and arithmetic_valid)
    request = CircuitTracerRequest(
        model_name="google/gemma-2-2b",
        transcoder_set="mntss/clt-gemma-2-2b-426k",
        prompt="0 + 5 * 3 + 5 =",
        backend="transformerlens",
        max_feature_nodes=64,
        batch_size=16,
    )
    result = run_circuit_tracer(request, upstream_attribution_valid=upstream_valid)
    return {
        "milestone": "6B_circuit_tracer_gate",
        "upstream_attribution_valid": upstream_valid,
        "cifar_all_counterfactual_valid": cifar_valid,
        "arithmetic_all_counterfactual_valid": arithmetic_valid,
        "resource_context": {
            "cuda_available": torch.cuda.is_available(),
            "pretrained_model_or_transcoder_execution_attempted": False,
            "note": "No compatible local pretrained model/transcoder pair is present; upstream gates fail closed first.",
        },
        "result": result,
        "claim": "tested fail-closed adapter only; no pretrained circuit-tracer attribution or intervention result",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cifar",
        type=Path,
        default=Path("research/cmae/results/counterfactual_assurance_m6b.json"),
    )
    parser.add_argument(
        "--arithmetic",
        type=Path,
        default=Path("research/cmae/results/arithmetic_fixed_target_m6b.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/cmae/results/circuit_tracer_gate_m6b.json"),
    )
    args = parser.parse_args()
    result = run_gate(args.cifar, args.arithmetic)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; status={result['result']['status']}")


if __name__ == "__main__":
    main()
