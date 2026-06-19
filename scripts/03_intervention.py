import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch

from src.config import Config
from src.intervention import OperatorAblator
from src.model_loader import MambaLoader
from src.static_decomp import StaticDecomposer


def main():
    config = Config()
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    loader = MambaLoader(config)
    decomposer = StaticDecomposer(loader, config)
    ablator = OperatorAblator(decomposer)
    vocab = loader.model.config.vocab_size
    batch = torch.randint(0, vocab, (config.batch, config.max_len), device=config.device)
    layer = list(decomposer.components.keys())[0]
    deltas = {str(i): ablator.ablate_component(batch, layer, i) for i in range(config.n_components)}
    with open(config.metrics_dir / "intervention.json", "w") as f:
        json.dump({"layer": layer, "ablation_delta_by_component": deltas}, f, indent=2)


if __name__ == "__main__":
    main()