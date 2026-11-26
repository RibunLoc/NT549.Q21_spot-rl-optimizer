"""
Export trained DQN (factored hoặc poolwise) to ONNX for Go controller.

Usage:
    python scripts/export_model.py --model results/kaggle/best_mixed_v23.pth \
        --output ../spot-rl-controller/models/dqn_v23.onnx
"""

import argparse
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_agent import _build_qnet
from envs.instance_catalog import STATE_DIM
from envs.action_schema import N_ACTIONS


def load_network(checkpoint_path: str, device: str = "cpu") -> torch.nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = ckpt.get("q_network_state_dict", ckpt)

    network_type = ckpt.get("network_type", "factored")
    hidden_dim = ckpt.get("hidden_dim")
    if hidden_dim is None and "trunk.0.weight" in sd:
        hidden_dim = sd["trunk.0.weight"].shape[0]
    hidden_dim = hidden_dim or 512

    state_dim = ckpt.get("state_dim", STATE_DIM)

    # factored (obs v1, 92 dims) hoặc poolwise (obs v2, 254 dims — controller Go cần cập nhật)
    net = _build_qnet(network_type, state_dim, ckpt.get("action_dim", N_ACTIONS), hidden_dim)
    net.load_state_dict(sd)
    net.eval()
    net.state_dim = state_dim
    print(f"  Network: {network_type} | state={state_dim} actions={net.n_actions} hidden={hidden_dim}")
    print(f"  Params: {sum(p.numel() for p in net.parameters()):,}")
    print(f"  Epsilon: {ckpt.get('epsilon', 'N/A')}, Steps: {ckpt.get('steps_done', 'N/A')}")
    return net


def export_onnx(net: torch.nn.Module, out_path: str):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dummy = torch.zeros(1, net.state_dim, dtype=torch.float32)
    torch.onnx.export(
        net, dummy, out_path,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["state"], output_names=["q_values"],
        dynamic_axes={"state": {0: "batch_size"}, "q_values": {0: "batch_size"}},
    )
    print(f"  Saved: {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")


def verify(net: torch.nn.Module, out_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnxruntime not installed")
        return

    test_input = np.random.rand(1, net.state_dim).astype(np.float32)
    with torch.no_grad():
        torch_out = net(torch.from_numpy(test_input)).numpy()
    sess = ort.InferenceSession(out_path)
    ort_out = sess.run(["q_values"], {"state": test_input})[0]
    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    print(f"  Max diff PyTorch vs ONNX: {max_diff:.2e}")
    assert max_diff < 1e-4, f"diff too large: {max_diff}"
    print(f"  Output shape: {ort_out.shape}")
    print(f"  Best action (random state): {int(np.argmax(ort_out[0]))} / {N_ACTIONS}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()

    print(f"Input : {args.model}")
    print(f"Output: {args.output}")
    print(f"\n[1/3] Loading checkpoint...")
    net = load_network(args.model)
    print(f"\n[2/3] Exporting ONNX (opset 17)...")
    export_onnx(net, args.output)
    if not args.no_verify:
        print(f"\n[3/3] Verifying...")
        verify(net, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
