import numpy as np
import torch

from research.cmae.backdoorbench_organism import CIFARBadNetsMLP
from research.cmae.binary_counterfactual import capture_binary_ggn
from research.cmae.binary_full_hessian import (
    capture_binary_full_hessian,
    damped_full_hessian_influence,
)


def test_full_hessian_matches_autograd_including_tanh_residual_terms() -> None:
    torch.manual_seed(12)
    model = CIFARBadNetsMLP(hidden_dim=2).double()
    inputs = torch.randn(3, 64, dtype=torch.float64) / 4
    labels = torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64)
    query = torch.randn(2, 64, dtype=torch.float64) / 4
    ggn = capture_binary_ggn(model, inputs, labels, query)
    artifact = capture_binary_full_hessian(model, inputs, labels, ggn)
    parameters = tuple(model.parameters())

    def objective(*values: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(torch.einsum("hd,nd->nh", values[0], inputs))
        logits = torch.einsum("h,nh->n", values[1][0], hidden)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)

    blocks = torch.autograd.functional.hessian(objective, parameters)
    sizes = [parameter.numel() for parameter in parameters]
    expected = torch.cat(
        [
            torch.cat(
                [blocks[row][column].reshape(sizes[row], sizes[column]) for column in range(2)],
                dim=1,
            )
            for row in range(2)
        ],
        dim=0,
    )
    torch.testing.assert_close(artifact.hessian, expected, rtol=1e-9, atol=1e-10)
    assert not torch.allclose(artifact.hessian, ggn.curvature)
    assert artifact.symmetry_residual < 1e-12
    damping = max(1e-3, -artifact.minimum_eigenvalue + 1e-3)
    scores = damped_full_hessian_influence(artifact, damping=damping)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
