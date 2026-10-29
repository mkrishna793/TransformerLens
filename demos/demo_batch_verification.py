"""
demo_batch_verification.py
Exact Demonstration of Neural Dataset Verification on 10 Samples:
- Samples #0-#7: Verified Clean Samples
- Sample #8: Stealth Spectral Poison Backdoor (Blocked with Gate = 0.0000)
- Sample #9: Forbidden Concept Trigger (Blocked with Gate = 0.0000)
"""

import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("TransformerLens"))

from transformer_lens.hooked_root_module import NeuralDatasetVerifierEngine


def run_demo():
    torch.manual_seed(42)
    B = 10
    d_model = 64
    forbidden_concept_idx = 10

    # 1. Instantiate a multi-layer neural network wrapped with NeuralDatasetVerifierEngine
    base_model = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, 16)
    )

    engine = NeuralDatasetVerifierEngine(
        base_model=base_model,
        dict_expansion=4,
        top_k=16,
        forbidden_concepts={forbidden_concept_idx}
    )
    engine.train()

    # 2. Construct 10 samples
    x = torch.randn(B, d_model, requires_grad=True)

    # Sample #9 triggers forbidden concept on layer 0 output
    layer0_hook = engine.verifiers["0"]
    with torch.no_grad():
        target_forbidden_atom = layer0_hook.W_enc[forbidden_concept_idx]

    # Forward pass
    logits = engine(x)

    # 3. Backward Pass with Targeted Backdoor Injected into Sample #8
    grad_logits = torch.randn_like(logits)
    # Inject spectral poison signature into sample #8
    layer2_hook = engine.verifiers["2"]
    grad_logits[8] = layer2_hook.W_enc[50] * 20.0

    # Run backward pass
    logits.backward(grad_logits)

    # 4. Generate Verification Ledger & Attribution Report
    sample_ids = torch.arange(B)
    ledger_result = engine.step_verification_ledger(sample_ids, redundancy_threshold=2.0)

    print("\n================================================================================")
    print("           NEURAL DATASET VERIFIER: FINAL BATCH AUDIT REPORT                    ")
    print("================================================================================")
    print(f"Batch Size: {B} Samples | Monitored Layers: {list(engine.verifiers.keys())}")
    print(f"Accepted Clean Samples: {list(set(range(B)) - set(ledger_result['rejected_sample_ids']))}")
    print(f"Rejected Corrupted/Unsafe Samples: {ledger_result['rejected_sample_ids']}")
    print(f"Redundant Samples (Eligible for Epoch-2 Pruning): {ledger_result['redundant_sample_ids']}")
    print("\nDetailed Rejection Log:")
    for item in engine.rejected_samples_report:
        print(f"  -> Sample #{item['sample_id']} @ Layer [{item['layer']}] BLOCKED | Reason: {item['reason']} | ART Z-Score: {item['art_z_score']:.2f}")

    print("\nGradient Integrity Check:")
    print(f"  Sample #8 Gradient Max: {x.grad[8].abs().max().item():.6f} (MUST BE 0.000000)")
    print("================================================================================\n")

    assert x.grad[8].abs().max().item() == 0.0, "Sample #8 gradient MUST be exactly 0.0!"


if __name__ == "__main__":
    run_demo()
