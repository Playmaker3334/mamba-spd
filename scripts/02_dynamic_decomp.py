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
    history = decomp.fit()
    pca = metrics.pca_baseline(decomp.descriptor)
    atoms, spec = metrics.atom_semantics(decomp, loader.tokenizer)
    coact = metrics.coactivation(decomp)
    summary = {
        "model_id": config.model_id,
        "layer": decomp.layer,
        "n_tokens": int(decomp.tokids.shape[0]),
        "descriptor_dim": int(decomp.descriptor.shape[1]),
        "block_dims": list(decomp.block_dims),
        "n_atoms": config.n_dynamic_atoms,
        "topk": config.topk,
        "block_balance": config.block_balance,
        "pca": pca,
        "sae": {"var_explained": round(decomp.fve, 4), "final_mse": history[-1]["mse"]},
        "specialization": spec,
    }
    json.dump(summary, open(config.out_dir / "run_summary.json", "w"), indent=2)
    json.dump(atoms, open(config.out_dir / "atoms.json", "w"), indent=2)
    json.dump(coact, open(config.out_dir / "coactivation.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("\n==== TOP 12 ATOMOS ====")
    for at in sorted([a for a in atoms if "top_tokens" in a], key=lambda x: -x["frequency"])[:12]:
        toks = ", ".join(f"{t['token']!r}:{t['lift']}" for t in at["top_tokens"][:5])
        print(f"  atomo {at['id']:2d} | freq {at['frequency']:.3f} | delta_z {at['delta_z']:+.2f} | pos {at['mean_position']:.2f} | {toks}")


if __name__ == "__main__":
    main()