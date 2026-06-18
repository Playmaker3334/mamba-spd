import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics
from src.config import Config
from src.generation import TextGenerator
from src.model_loader import MambaLoader
from src.static_decomp import StaticDecomposer

PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered",
    "def fibonacci(n):",
    "The meaning of life is",
]


def main():
    config = Config()
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    loader = MambaLoader(config)
    gen = TextGenerator(loader)
    decomposer = StaticDecomposer(loader, config)

    examples = []
    for p in PROMPTS:
        text = gen.generate(p)
        ppl = gen.perplexity(p + " " + text)
        ids = loader.encode(p)
        _, ci = decomposer.causal_importances(ids)
        examples.append({
            "prompt": p,
            "generation": text,
            "perplexity": ppl,
            "l0_per_token": metrics.l0_per_token(ci),
        })

    with open(config.metrics_dir / "generation_probe.json", "w") as f:
        json.dump({"model_id": config.model_id, "examples": examples}, f, indent=2)


if __name__ == "__main__":
    main()
