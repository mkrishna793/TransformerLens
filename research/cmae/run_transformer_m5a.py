"""Sequential Milestone-5A ground-truth Transformer circuit benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

from research.cmae.arithmetic_data import (
    MODULUS,
    ArithmeticSplit,
    make_arithmetic_dataset,
)
from research.cmae.arithmetic_transformer import (
    ArithmeticTransformer,
    evaluate_arithmetic_transformer,
    train_arithmetic_transformer,
)
from research.cmae.transformer_circuit import trace_component_circuit


def run_transformer_m5a(
    output: Path,
    *,
    seed: int = 101,
    epochs: int = 250,
    learning_rate: float = 0.005,
    quadrature_steps: int = 256,
) -> dict[str, object]:
    dataset = make_arithmetic_dataset(seed)
    torch.manual_seed(seed)
    model = ArithmeticTransformer()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    history = train_arithmetic_transformer(
        model,
        dataset,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed + 1,
    )
    split_names: tuple[ArithmeticSplit, ...] = (
        "train",
        "calibration",
        "iid",
        "composition",
        "operand",
        "multiplier",
        "combined",
    )
    evaluation = {
        split: evaluate_arithmetic_transformer(model, dataset, split) for split in split_names
    }
    accuracies = {split: float(evaluation[split]["accuracy"]) for split in split_names}
    generalization_summary = {
        "chance_accuracy": 1.0 / MODULUS,
        "train_minus_iid_accuracy": accuracies["train"] - accuracies["iid"],
        "iid_minus_composition_accuracy": accuracies["iid"] - accuracies["composition"],
        "iid_minus_operand_accuracy": accuracies["iid"] - accuracies["operand"],
        "iid_minus_multiplier_accuracy": accuracies["iid"] - accuracies["multiplier"],
        "iid_minus_combined_accuracy": accuracies["iid"] - accuracies["combined"],
    }
    traces: list[dict[str, object]] = []
    trace_splits: tuple[ArithmeticSplit, ...] = (
        "calibration",
        "iid",
        "composition",
        "operand",
        "multiplier",
        "combined",
    )
    for split in trace_splits:
        sample = dataset.split_samples(split)[0]
        traces.append(
            trace_component_circuit(model, sample, quadrature_steps=quadrature_steps).to_dict()
        )
    trace_metrics = [cast(dict[str, object], trace["metrics"]) for trace in traces]
    max_quadrature = max(
        abs(cast(float, metrics["integrated_quadrature_residual"])) for metrics in trace_metrics
    )
    max_reconstruction = max(
        cast(float, metrics["residual_reconstruction_error"]) for metrics in trace_metrics
    )
    max_parity = max(abs(cast(float, metrics["readout_parity_error"])) for metrics in trace_metrics)
    result: dict[str, object] = {
        "schema_version": 1,
        "milestone": "5A ground-truth Transformer circuit foundation",
        "execution": "sequential",
        "seed": seed,
        "task": {
            "expression": "u=(a+b) mod p; v=(u*c) mod p; y=(v+d) mod p",
            "modulus": MODULUS,
            "prompt": "BOS MOD p ( a + b ) * c + d =",
            "composition_holdout": "raw a+b=7 is excluded from train/calibration/IID",
            "counts": dataset.counts(),
            "all_output_tokens_covered_by_train": True,
        },
        "model": {
            "architecture": "2-layer pre-LN causal Transformer; 32 residual channels; 2 heads; GELU MLPs",
            "parameter_count": parameter_count,
            "hook_system": "TransformerLens v4 HookedRootModule and HookPoint",
            "pretrained": False,
        },
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "initial_loss": history[0],
            "final_loss": history[-1],
            "history": history,
        },
        "evaluation": evaluation,
        "generalization_summary": generalization_summary,
        "trace_summary": {
            "trace_count": len(traces),
            "nodes_per_trace": trace_metrics[0]["node_count"],
            "quadrature_steps": quadrature_steps,
            "max_abs_integrated_quadrature_residual": max_quadrature,
            "max_residual_reconstruction_error": max_reconstruction,
            "max_abs_readout_parity_error": max_parity,
            "integrated_readout_healthy": bool(
                max_quadrature < 1e-3 and max_reconstruction < 1e-4 and max_parity < 1e-4
            ),
            "mean_first_order_pearson": float(
                np.mean([cast(float, metrics["first_order_pearson"]) for metrics in trace_metrics])
            ),
            "mean_first_order_spearman": float(
                np.mean([cast(float, metrics["first_order_spearman"]) for metrics in trace_metrics])
            ),
            "mean_shuffled_pearson": float(
                np.mean(
                    [
                        cast(float, metrics["shuffled_prediction_mean_pearson"])
                        for metrics in trace_metrics
                    ]
                )
            ),
        },
        "traces": traces,
        "formal_verification": {
            "rocq_closed_theorems": 6,
            "z3_expected_unsat": 5,
            "z3_expected_sat_counterexamples": 1,
            "artifact": "formal_verification/rocq/CMAETransformerCircuitFoundations.v",
            "independent_checker": "formal_verification/verify_cmae_transformer_circuit_z3.py",
        },
        "limitations": [
            "A small locally trained modular-arithmetic Transformer, not a pretrained language model or Claude-scale reproduction.",
            "Component-to-logit measurements are not yet a full feature-to-feature CLT attribution graph.",
            "Integrated direct-readout allocation differs from total causal effect; downstream mediation is reported separately.",
            "Zero ablation is one intervention baseline and can send activations off-distribution.",
            "One seed and development holdouts; no confirmatory generalization or SAE/NLA semantic claim.",
        ],
    }
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--quadrature-steps", type=int, default=256)
    arguments = parser.parse_args()
    result = run_transformer_m5a(
        arguments.output,
        seed=arguments.seed,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        quadrature_steps=arguments.quadrature_steps,
    )
    print(json.dumps(result["trace_summary"], indent=2, allow_nan=False))
    print(json.dumps(result["evaluation"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
