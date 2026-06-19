import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics
from src.config import Config
from src.depth_profile import depth_profile
from src.dynamic_ops import build_corpus_blocks
from src.model_loader import MambaLoader


def main():
    config = Config()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    loader = MambaLoader(config)
    blocks = build_corpus_blocks(loader, config)
    layers = list(range(loader.model.config.num_hidden_layers))
    res = depth_profile(loader, blocks, config, metrics.categorize_token, layers=layers)
    json.dump(res, open(config.out_dir / "depth_profile.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\n==== Cohen's d vs word_initial POR CAPA ====")
    layers = res["layers_sampled"]
    print("  capa:        " + "  ".join(f"{l:5d}" for l in layers))
    for cat, byl in res["cohen_d_vs_baseline_by_layer"].items():
        print(f"  {cat:14} " + "  ".join(f"{byl[str(l)]:+.2f}" for l in layers))


if __name__ == "__main__":
    main()