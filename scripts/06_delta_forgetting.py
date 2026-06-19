import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics
from src.config import Config
from src.dynamic_ops import DynamicOperatorDictionary, build_corpus_blocks
from src.model_loader import MambaLoader


def main():
    config = Config()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    loader = MambaLoader(config)
    blocks = build_corpus_blocks(loader, config)
    decomp = DynamicOperatorDictionary(config)
    decomp.collect(loader, blocks)
    res = metrics.delta_by_category(decomp, loader.tokenizer)
    json.dump(res, open(config.out_dir / "delta_forgetting.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\n==== Δ POR CATEGORIA (orden ascendente) ====")
    for r in res["categories"]:
        cd = r.get("cohen_d_vs_word_initial", 0.0)
        print(f"  {r['category']:14} | n {r['n_tokens']:6d} | Δ {r['delta_mean']:.5f} | z {r['z_vs_global']:+.2f} | d_vs_word {cd:+.2f}")
    print(f"\n  corr(log-freq, Δ) = {res['corr_logfreq_delta']:+.3f}")


if __name__ == "__main__":
    main()