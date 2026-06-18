import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src import metrics
from src.config import Config
from src.dynamic_ablation import AblationProbe, select_target_atom
from src.dynamic_ops import DynamicOperatorDictionary, build_corpus_blocks
from src.model_loader import MambaLoader


def build_eval_blocks(loader, config, n_blocks=400):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    ids = []
    for row in ds:
        t = row["text"].strip()
        if len(t) < 20:
            continue
        ids.extend(loader.tokenizer(t, add_special_tokens=False).input_ids)
        if len(ids) >= n_blocks * config.max_len + config.max_len:
            break
    return torch.tensor(ids[:n_blocks * config.max_len]).reshape(n_blocks, config.max_len)


def main():
    torch.manual_seed(0)
    config = Config()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    loader = MambaLoader(config)
    blocks = build_corpus_blocks(loader, config)
    decomp = DynamicOperatorDictionary(config)
    decomp.collect(loader, blocks)
    decomp.fit()
    atoms, _ = metrics.atom_semantics(decomp, loader.tokenizer)
    atom, sig = select_target_atom(atoms, mode="max_delta_z")
    print(f"atomo objetivo (max delta_z, seleccion automatica): {atom}")
    print("  firma:", ", ".join(f"{t['token']!r}:{t['lift']}" for t in sig["top_tokens"][:6]))
    print(f"  delta_z {sig['delta_z']:+.2f} | freq {sig['frequency']:.3f} | max_lift {sig['max_lift']}")

    eval_blocks = build_eval_blocks(loader, config)
    print(f"eval: {tuple(eval_blocks.shape)} (held-out validation) | corriendo ablacion...")
    probe = AblationProbe(decomp, loader)
    results = probe.run(eval_blocks, atom)
    results["atom_signature"] = [{"token": t["token"], "lift": t["lift"]} for t in sig["top_tokens"][:8]]
    results["atom_delta_z"] = sig["delta_z"]
    results["selection_mode"] = "max_delta_z"

    json.dump(results, open(config.out_dir / "ablation.json", "w"), indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()