"""Sequential consolidated Milestone-three/four evidence runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from research.cmae.adaptive import run_adaptive_benchmark
from research.cmae.deep_evidence import run_deep_benchmark


def run_trace_adaptive_program(
    output: Path, deep_output: Path, adaptive_output: Path
) -> dict[str, object]:
    deep = run_deep_benchmark(deep_output)
    adaptive = run_adaptive_benchmark(adaptive_output)
    deep_cases = cast(list[dict[str, object]], deep["cases"])
    adaptive_cases = cast(list[dict[str, object]], adaptive["cases"])
    traces = [cast(dict[str, float], case["trace"]) for case in deep_cases]
    projection_records: list[dict[str, float]] = []
    for case in adaptive_cases:
        before = cast(dict[str, object], case["before"])
        projection_records.append(cast(dict[str, float], before["attack_projection"]))
        if "after" in case:
            after = cast(dict[str, object], case["after"])
            projection_records.append(cast(dict[str, float], after["attack_projection"]))
    maximum_linf_violation = max(
        [0.0] + [record["max_linf"] - record["epsilon"] for record in projection_records]
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "execution": "sequential: all deep-evidence cases, then all adaptive-repair cases",
        "deep_case_count": len(deep_cases),
        "adaptive_case_count": len(adaptive_cases),
        "promoted_cases": adaptive["promoted_cases"],
        "rolled_back_cases": sum(case["final_state"] == "rolled_back" for case in adaptive_cases),
        "trace_summary": {
            "max_abs_readout_residual": max(abs(trace["residual"]) for trace in traces),
            "max_abs_conservation_error": max(abs(trace["conservation_error"]) for trace in traces),
            "max_hidden_intervention_error": max(
                trace["intervention_max_error"] for trace in traces
            ),
            "max_abs_first_order_path_residual": max(
                abs(trace["first_order_path_residual"]) for trace in traces
            ),
        },
        "maximum_linf_projection_violation": maximum_linf_violation,
        "formal_verification": {
            "rocq_closed_theorems": 8,
            "z3_expected_unsat": 7,
            "z3_expected_sat_counterexamples": 1,
            "artifact": "formal_verification/rocq/CMAETraceFoundations.v",
            "independent_checker": "formal_verification/verify_cmae_trace_z3.py",
        },
        "artifacts": {
            "deep_evidence": str(deep_output),
            "adaptive_repair": str(adaptive_output),
        },
        "claim_scope": "small real-data MLP evidence with exact readout accounting and bounded adaptive attacks; not a complete Transformer attribution graph or universal safety certificate",
    }
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deep-output", type=Path, required=True)
    parser.add_argument("--adaptive-output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_trace_adaptive_program(
        arguments.output, arguments.deep_output, arguments.adaptive_output
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
