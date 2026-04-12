"""
Export trained DQN model (.pth) to ONNX format for use in Go controller.

Usage:
    python scripts/export_model.py --model results/models/dqn_stable_v5_best.pth --output models/dqn_stable.onnx
    python scripts/export_model.py --model results/models/dqn_volatile_v5_best.pth --output models/dqn_volatile.onnx
    python scripts/export_model.py --model results/models/dqn_spike_v5_best.pth --output models/dqn_spike.onnx
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.networks import QNetwork


# === Constants (must match spot_env.py) ===
STATE_DIM = 15   # 15 observation features
ACTION_DIM = 7   # 7 discrete actions

ACTION_NAMES = [
    "REQUEST_SPOT",        # 0
    "REQUEST_ONDEMAND",    # 1
    "TERMINATE_SPOT",      # 2
    "TERMINATE_ONDEMAND",  # 3
    "MIGRATE_TO_ONDEMAND", # 4
    "MIGRATE_TO_SPOT",     # 5
    "DO_NOTHING",          # 6
]


def load_qnetwork(checkpoint_path: str, device: str = "cpu") -> QNetwork:
    """Load QNetwork weights from DQNAgent checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Checkpoint có thể là DQNAgent checkpoint (có q_network_state_dict)
    # hoặc là QNetwork state_dict trực tiếp
    if "q_network_state_dict" in checkpoint:
        state_dict = checkpoint["q_network_state_dict"]
        print(f"  Loaded from DQNAgent checkpoint")
        print(f"  epsilon={checkpoint.get('epsilon', 'N/A'):.4f}, "
              f"steps={checkpoint.get('steps_done', 'N/A')}")
    else:
        state_dict = checkpoint
        print(f"  Loaded as raw state_dict")

    model = QNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(model: QNetwork, output_path: str):
    """Export QNetwork to ONNX format."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Dummy input: batch_size=1, state_dim=15, tất cả normalized về [0,1]
    dummy_input = torch.zeros(1, STATE_DIM, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["state"],      # tên input node trong ONNX graph
        output_names=["q_values"],  # tên output node
        dynamic_axes={
            "state": {0: "batch_size"},       # batch dim là dynamic
            "q_values": {0: "batch_size"},
        },
    )
    print(f"  ONNX model saved: {output_path}")


def verify_onnx(model: QNetwork, output_path: str):
    """Verify ONNX output matches PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed, skipping verification")
        print("         Install with: pip install onnxruntime")
        return

    # Random state input
    test_input = np.random.rand(1, STATE_DIM).astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        torch_out = model(torch.FloatTensor(test_input)).numpy()

    # ONNX Runtime inference
    sess = ort.InferenceSession(output_path)
    ort_out = sess.run(["q_values"], {"state": test_input})[0]

    # Compare
    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"  Max diff PyTorch vs ONNX: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Outputs differ too much: {max_diff}"
    print(f"  Verification PASSED")

    # Show sample Q-values
    print(f"\n  Sample Q-values for random state:")
    best_action = np.argmax(ort_out[0])
    for i, (name, q) in enumerate(zip(ACTION_NAMES, ort_out[0])):
        marker = " <-- BEST" if i == best_action else ""
        print(f"    {i}: {name:<25} Q={q:+.4f}{marker}")


def print_model_info(model: QNetwork):
    """Print model architecture summary."""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Architecture: {model.network}")
    print(f"  Total params: {total_params:,}")
    print(f"  Input  dim: {STATE_DIM} (normalized state vector)")
    print(f"  Output dim: {ACTION_DIM} (Q-value per action)")


def main():
    parser = argparse.ArgumentParser(description="Export DQN model to ONNX")
    parser.add_argument(
        "--model", required=True,
        help="Path to .pth checkpoint file"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for .onnx file"
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip ONNX verification step"
    )
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"Exporting DQN model to ONNX")
    print(f"{'='*50}")
    print(f"  Input : {args.model}")
    print(f"  Output: {args.output}")

    # 1. Load model
    print(f"\n[1/3] Loading checkpoint...")
    model = load_qnetwork(args.model)
    print_model_info(model)

    # 2. Export
    print(f"\n[2/3] Exporting to ONNX (opset 17)...")
    export_onnx(model, args.output)

    # 3. Verify
    if not args.no_verify:
        print(f"\n[3/3] Verifying ONNX output...")
        verify_onnx(model, args.output)
    else:
        print(f"\n[3/3] Verification skipped")

    # Print file size
    size_kb = os.path.getsize(args.output) / 1024
    print(f"\n{'='*50}")
    print(f"Done! File size: {size_kb:.1f} KB")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
