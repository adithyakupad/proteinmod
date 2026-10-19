"""Same-environment differential kernels/training and optional archived M17 state replay.

Run in separate processes with baseline/current src first on PYTHONPATH. Reads no
M18 target files; writes only the explicitly supplied new output directory.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from proteinmod.data import write_json
from proteinmod.hybrid import CrossAttention
from proteinmod.lora_model import AdaptedESM, tensor_digest
from proteinmod.mechanism import degree_null
from proteinmod.structural_gnn import ResidueGNN, train_gnn


def load_module(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def digest_array(value):
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def kernels(output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fixture = load_module("tests/test_structural_gnn.py", "graph_fixture")
    lora_fixture = load_module("tests/test_lora_neural.py", "lora_fixture")
    graph = fixture.tiny()
    torch.manual_seed(23)
    dynamic = torch.zeros(2, 3, 25)
    dynamic[0, 0, 0] = 1
    dynamic[1, 1:, 0] = 1
    gnn = ResidueGNN(**graph, hidden=4, layers=2).eval()
    graph_prediction = gnn(dynamic)
    graph_prediction.square().sum().backward()
    graph_gradients = {name: value.grad for name, value in gnn.named_parameters()}
    attention = CrossAttention(4, 2).eval()
    query = torch.randn(2, 2, 4)
    context = torch.randn(2, 3, 4)
    query_mask = torch.tensor([[True, False], [True, True]])
    key_mask = torch.tensor([[True, True, False], [True, True, True]])
    attended, weights = attention(query, context, query_mask, key_mask)
    base, settings = lora_fixture.tiny()
    adapted = AdaptedESM(base, settings, hidden=4, seed=0)
    prefix = torch.randn(3, 6, 12, requires_grad=True)
    extended_mask = torch.zeros(3, 1, 1, 6)
    mutation_mask = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 0]])
    adapted_prediction = adapted(
        prefix=prefix, extended_mask=extended_mask, mutation_mask=mutation_mask
    )
    adapted_prediction.square().sum().backward()
    gradients = {
        name: value.grad for name, value in adapted.named_parameters() if value.requires_grad
    }
    assert all(value is not None for value in gradients.values())
    assert prefix.grad[-1].abs().sum() > 0
    frozen_before = adapted.frozen_digest()
    optimizer = torch.optim.Adam(
        [value for value in adapted.parameters() if value.requires_grad], lr=0.01
    )
    optimizer.step()
    assert adapted.frozen_digest() == frozen_before
    checkpoint = adapted.selected_state()
    restored = AdaptedESM(base, settings, hidden=4, seed=0)
    restored.load_selected(checkpoint)
    with torch.no_grad():
        expected = adapted(prefix=prefix, extended_mask=extended_mask, mutation_mask=mutation_mask)
        actual = restored(prefix=prefix, extended_mask=extended_mask, mutation_mask=mutation_mask)
    assert torch.equal(expected, actual)
    inputs = np.zeros((8, 3, 25), np.float32)
    inputs[:, 0, 0] = 1
    inputs[:, 0, 1] = np.arange(8) / 8
    targets = np.arange(8, dtype=float) / 10
    training = {
        "device": "cpu",
        "cpu_threads": 1,
        "hidden": 4,
        "learning_rate": 0.01,
        "weight_decays": [0.01],
        "batch_size": 2,
        "max_epochs": 3,
        "patience": 2,
    }
    result, prediction, state = train_gnn(
        graph, inputs, targets, np.arange(4), np.array([4, 5]), np.array([6, 7]), training, 0, 2
    )
    result.pop("elapsed_seconds")
    null_edges, swaps = degree_null(graph["edge_index"], 31)
    write_json(
        output / "kernels.json",
        {
            "gnn_prediction": digest_array(graph_prediction.detach().numpy()),
            "gnn_gradients": tensor_digest(graph_gradients),
            "attention_prediction": digest_array(attended.detach().numpy()),
            "attention_weights": digest_array(weights.detach().numpy()),
            "lora_prediction": digest_array(adapted_prediction.detach().numpy()),
            "lora_gradients": tensor_digest(gradients),
            "WT_and_mutant_prefix_gradients": digest_array(prefix.grad.numpy()),
            "lora_selected_state": tensor_digest(checkpoint),
            "frozen_backbone": frozen_before,
            "restored_prediction": digest_array(actual.numpy()),
            "training_trace_selection_metrics": result,
            "trained_prediction": digest_array(prediction),
            "trained_state": tensor_digest(state["state_dict"]),
            "degree_null": digest_array(null_edges),
            "degree_swaps": swaps,
            "torch_RNG_after": digest_array(torch.random.get_rng_state().numpy()),
        },
    )
    durations = []
    for _ in range(7):
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(300):
                gnn(dynamic)
        durations.append(time.perf_counter() - started)
    write_json(
        output / "performance.json",
        {
            "gnn_300_forward_seconds": durations,
            "median_seconds": float(np.median(durations)),
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "device": "CPU",
            "threads": 1,
            "scope": "Tiny GNN timing; process peak includes imports/fixtures. Not a production training memory benchmark.",
        },
    )


def m17(output):
    module = load_module("scripts/verify_m17.py", "m17_replay")
    runs = json.loads(Path("benchmarks/m17/runs.json").read_text())
    cases = json.loads(Path("benchmarks/m17/split-manifest.json").read_text())["folds"]
    grouped = defaultdict(list)
    with Path("benchmarks/m17/predictions.csv").open() as handle:
        for row in csv.DictReader(handle):
            key = tuple(
                str(row[name]) for name in ("study", "fold", "protocol", "seed", "fraction")
            )
            grouped[key + (row["model"],)].append(row)

    # Redirect only the replay report; original prediction/state checks remain unchanged.
    def write_replay_report(path, value):
        if path != "runs/m17/state-replay.json":
            raise ValueError("Unexpected replay report destination")
        write_json(output / "m17.json", value)

    module.write_json = write_replay_report
    module.replay(runs, cases, grouped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m17", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    kernels(args.output)
    if args.m17:
        m17(args.output)
    write_json(
        args.output / "execution.json",
        {
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "data_outcomes": "Only synthetic targets and optional previously published M17 development labels; no M18 external outcomes.",
        },
    )
    print(args.output)
