import json
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src import metrics
from src.config import Config
from src.dynamic_ablation import AblationProbe, select_target_atom
from src.dynamic_ops import DynamicOperatorDictionary, build_corpus_blocks
from src.model_loader import MambaLoader

N_CONTROLS = 8
CONTROL_MAX_ABS_Z = 0.5


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


def select_control_atoms(atoms, target_sig, n_controls=N_CONTROLS, max_abs_z=CONTROL_MAX_ABS_Z):
    target_id = target_sig["id"]
    target_freq = target_sig["frequency"]
    candidates = [
        a for a in atoms
        if "delta_z" in a
        and a["id"] != target_id
        and abs(a["delta_z"]) <= max_abs_z
        and a["frequency"] > 0
    ]
    candidates.sort(key=lambda a: abs(a["frequency"] - target_freq))
    return candidates[:n_controls]


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

    controls = select_control_atoms(atoms, sig)
    print(f"\ncontrol: {len(controls)} atomos pareados por frecuencia (~{sig['frequency']:.3f}) con |delta_z| <= {CONTROL_MAX_ABS_Z}")
    control_rows = []
    for csig in controls:
        cid = csig["id"]
        cres = probe.run(eval_blocks, cid)
        row = {
            "atom": cid,
            "frequency": csig["frequency"],
            "delta_z": csig["delta_z"],
            "specificity_ratio": cres["targeted"]["specificity_ratio"],
            "kl_on_atom_tokens": cres["targeted"]["kl_on_atom_tokens"],
            "kl_off_atom_tokens": cres["targeted"]["kl_off_atom_tokens"],
            "n_atom_active": cres["targeted"]["n_atom_active"],
        }
        control_rows.append(row)
        print(f"  control {cid:2d} | freq {csig['frequency']:.3f} | dz {csig['delta_z']:+.2f} | spec_ratio {row['specificity_ratio']:.3f}")

    target_ratio = results["targeted"]["specificity_ratio"]
    if control_rows:
        ratios = [r["specificity_ratio"] for r in control_rows]
        null_mean = round(statistics.mean(ratios), 3)
        null_median = round(statistics.median(ratios), 3)
        null_std = round(statistics.pstdev(ratios), 3) if len(ratios) > 1 else 0.0
        null_max = round(max(ratios), 3)
        n_ge = sum(1 for r in ratios if r >= target_ratio)
    else:
        null_mean = null_median = null_std = null_max = None
        n_ge = None

    results["control"] = {
        "n_controls": len(control_rows),
        "selection": {"matched_on": "frequency", "max_abs_delta_z": CONTROL_MAX_ABS_Z},
        "target_specificity_ratio": target_ratio,
        "control_specificity_ratio_mean": null_mean,
        "control_specificity_ratio_median": null_median,
        "control_specificity_ratio_std": null_std,
        "control_specificity_ratio_max": null_max,
        "n_controls_ge_target": n_ge,
        "controls": control_rows,
    }

    json.dump(results, open(config.out_dir / "ablation.json", "w"), indent=2)
    print("\n" + json.dumps(results, indent=2))

    print("\n==== CONTROL: nulo de specificity_ratio ====")
    print(f"  objetivo (atomo {atom}, delta_z {sig['delta_z']:+.2f}): {target_ratio:.3f}")
    if control_rows:
        print(f"  control n={len(control_rows)} | media {null_mean} | mediana {null_median} | std {null_std} | max {null_max}")
        print(f"  controles con spec_ratio >= objetivo: {n_ge}/{len(control_rows)}")
    else:
        print("  sin atomos de control que cumplan el criterio; ampliar CONTROL_MAX_ABS_Z o N_CONTROLS")


if __name__ == "__main__":
    main()