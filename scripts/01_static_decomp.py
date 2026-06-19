import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from src import metrics
from src.config import Config
from src.model_loader import MambaLoader
from src.static_decomp import StaticDecomposer


def make_sampler(loader, config):
    vocab = loader.model.config.vocab_size

    def sample():
        return torch.randint(0, vocab, (config.batch, config.max_len), device=config.device)

    return sample


def main():
    config = Config()
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    loader = MambaLoader(config)
    decomposer = StaticDecomposer(loader, config)
    sampler = make_sampler(loader, config)
    history = decomposer.train(sampler)
    batch = sampler()
    _, ci = decomposer.causal_importances(batch)
    result = {
        "model_id": config.model_id,
        "n_components": config.n_components,
        "n_layers_decomposed": len(decomposer.components),
        "final": history[-1],
        "l0_per_token": metrics.l0_per_token(ci),
        "history": history,
    }
    with open(config.metrics_dir / "static_decomp.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()