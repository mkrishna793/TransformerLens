import numpy as np
import torch

from research.cmae.backdoorbench_organism import CIFARBatch, build_cifar_model
from research.cmae.curvature_influence import RawSampleGradient
from research.cmae.run_counterfactual_assurance import (
    AssuranceConfig,
    AttributionRun,
    _counterfactual_gate,
    _detection_metrics,
    _physical_repair,
)


def test_fixed_budget_detection_accounting() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    metrics = _detection_metrics(scores, labels, (0.25, 0.5))
    assert metrics["roc_auc"] == 1.0
    assert metrics["budgets"]["0.25"]["quarantine_count"] == 1
    assert metrics["budgets"]["0.25"]["recall"] == 0.5
    assert metrics["budgets"]["0.25"]["clean_false_positive_rate"] == 0.0
    assert metrics["budgets"]["0.5"]["recall"] == 1.0


def test_failed_counterfactual_gate_blocks_physical_repair() -> None:
    gate = _counterfactual_gate(
        {
            "spearman": 0.2,
            "sign_agreement": 0.5,
            "pearson": 0.0,
            "mean_absolute_error_raw_scale": 1.0,
            "actual_mean_absolute_effect": 1.0,
        }
    )
    assert not gate["pass"]
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    batch = CIFARBatch(
        inputs=torch.zeros(4, 64, dtype=torch.float64),
        labels=torch.from_numpy(labels).double(),
        original_labels=labels,
        poison=np.array([0, 0, 1, 1], dtype=np.int64),
        sample_ids=np.arange(4, dtype=np.int64),
    )
    attribution = AttributionRun(
        scores={"exact_ggn": np.zeros(4), "random": np.zeros(4), "training_loss": np.zeros(4)},
        raw_samples=(),
        raw_query=RawSampleGradient((0, 0, 0, -1), {}),
    )
    repair = _physical_repair(
        build_cifar_model(1, hidden_dim=2),
        batch,
        batch,
        torch.zeros(2, 64, dtype=torch.float64),
        attribution,
        {"standard": {"attack_success_rate": 1.0, "clean_accuracy": 1.0}},
        seed=1,
        config=AssuranceConfig(train_count=4, epochs=1),
        admissible=False,
        upstream={"counterfactual": gate},
    )
    assert repair["status"] == "blocked"
