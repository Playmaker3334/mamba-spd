import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from src.config import Config
from src.dynamic_ops import DynamicOperatorDictionary
from src.model_loader import MambaLoader

CORPUS = [
    "The cat sat on the mat and looked outside.",
    "In 1969 humans first walked on the surface of the moon.",
    "def factorial(n): return 1 if n == 0 else n * factorial(n - 1)",
    "She opened the door slowly, afraid of what she might find.",
    "The stock market fell sharply after the announcement was made.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "He repeated the same word again and again and again and again.",
    "The capital of France is Paris and the capital of Japan is Tokyo.",
]


def main():
    config = Config()
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    loader = MambaLoader(config)
    layer_idx = loader.model.config.num_hidden_layers // 2
    corpus = [loader.encode(t) for t in CORPUS]

    dyn = DynamicOperatorDictionary(config)
    operators = dyn.collect_operators(loader, corpus, layer_idx).to(config.device)
    history = dyn.fit(operators)
    usage = dyn.atom_usage(operators)

    result = {
        "layer": layer_idx,
        "n_tokens": operators.shape[0],
        "d_op": operators.shape[1],
        "n_atoms": config.n_dynamic_atoms,
        "final": history[-1],
        "atom_usage": usage,
        "history": history,
    }
    with open(config.metrics_dir / "dynamic_decomp.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()