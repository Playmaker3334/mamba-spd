import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Config
from src.dynamic_ops import build_corpus_blocks
from src.metrics import categorize_token
from src.model_loader import MambaLoader
from src import implicit_knockout as ik

HORIZON = 32


def kl_rows(p_logits, q_logits):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.log_softmax(q_logits, dim=-1)
    return (p.exp() * (p - q)).sum(-1)


def build_eval_blocks(loader, config, n_blocks=200):
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
    config = Config()
    layer = config.layer if config.layer is not None else None

    loader = MambaLoader(config)
    model = loader.model
    tok = loader.tokenizer
    n_layers = model.config.num_hidden_layers
    if layer is None:
        layer = n_layers // 2

    train_blocks = build_corpus_blocks(loader, config)
    eval_blocks = build_eval_blocks(loader, config, n_blocks=200)
    n_eval = eval_blocks.shape[0]
    L = eval_blocks.shape[1]

    if HORIZON >= L:
        raise ValueError(f"HORIZON ({HORIZON}) debe ser menor que seq_len ({L})")

    V = model.config.vocab_size
    raws = tok.convert_ids_to_tokens(list(range(V)))
    cat_of_id = [categorize_token(r) if r is not None else "other" for r in raws]
    counts = torch.bincount(train_blocks.reshape(-1), minlength=V).float()

    ik.install()

    sample = eval_blocks[:1].to(config.device)
    with torch.no_grad():
        native = model(sample).logits.float()
    recon = ik.logits_for(model, sample, layer, mask=None)
    validity_max_abs = float((recon - native).abs().max())

    max_pairs_per_block = 3
    max_pos = L - HORIZON

    records = []
    for bi in range(n_eval):
        block = eval_blocks[bi:bi + 1].to(config.device)
        ids = eval_blocks[bi].tolist()
        cats = [cat_of_id[t] for t in ids]
        num_pos = [p for p in range(max_pos) if cats[p] == "number"]
        word_pos = [p for p in range(max_pos) if cats[p] == "word_initial"]
        if not num_pos or not word_pos:
            continue

        base = ik.logits_for(model, block, layer, mask=None)

        pairs = []
        used_w = set()
        for p in num_pos[:max_pairs_per_block]:
            cand = sorted(word_pos, key=lambda w: abs(w - p))
            wsel = next((w for w in cand if w not in used_w), None)
            if wsel is None:
                continue
            used_w.add(wsel)
            pairs.append((p, wsel))

        for (pn, pw) in pairs:
            for (pos, role) in [(pn, "number"), (pw, "word_initial")]:
                mask = torch.zeros(1, L, dtype=torch.bool, device=config.device)
                mask[0, pos] = True
                ko = ik.logits_for(model, block, layer, mask=mask)
                kl = kl_rows(base[0], ko[0])
                curve = kl[pos + 1:pos + 1 + HORIZON].detach().cpu()
                records.append({
                    "block": bi,
                    "role": role,
                    "position": pos,
                    "token_id": ids[pos],
                    "curve": [round(float(x), 8) for x in curve],
                })

    ik.uninstall()

    def mean_curve(role):
        rs = [r for r in records if r["role"] == role]
        if not rs:
            return None, 0
        M = torch.tensor([r["curve"] for r in rs])
        return M.mean(0), len(rs)

    def pos_mean(role):
        ps = [r["position"] for r in records if r["role"] == role]
        return round(sum(ps) / len(ps), 2) if ps else None

    num_curve, n_num = mean_curve("number")
    word_curve, n_word = mean_curve("word_initial")

    out = {
        "design": "implicit_knockout_decay_profile_by_distance",
        "prediction": "number knockout KL decays faster with distance than content-words: starts higher at offset 1 and crosses below within HORIZON",
        "preregistered": True,
        "eval_split": "wikitext-2-raw-v1 validation (held-out); frequency reference from train split",
        "horizon": HORIZON,
        "position_filter": f"only knockouts with full HORIZON of future (pos <= {max_pos - 1})",
        "pairing": "matched by position only (nearest word_initial); frequency decoupling established in 06 (corr logfreq-delta = -0.014)",
        "confirm_if": "number_curve[offset=1] > word_curve[offset=1] (starts higher) AND number far-region mean < word far-region mean (ends lower) => crossover within horizon",
        "layer": layer,
        "n_layers": n_layers,
        "n_eval_blocks": n_eval,
        "seq_len": L,
        "validity_check_logits_max_abs_recon_minus_native": round(validity_max_abs, 7),
        "n_number": n_num,
        "n_word_initial": n_word,
        "position_mean_number": pos_mean("number"),
        "position_mean_word_initial": pos_mean("word_initial"),
    }

    if num_curve is not None and word_curve is not None:
        offsets = list(range(1, HORIZON + 1))
        far = slice(HORIZON // 2, HORIZON)

        num_near = float(num_curve[0])
        word_near = float(word_curve[0])
        num_far = float(num_curve[far].mean())
        word_far = float(word_curve[far].mean())

        num_norm = num_curve / (num_curve[0] + 1e-12)
        word_norm = word_curve / (word_curve[0] + 1e-12)
        num_norm_far = float(num_norm[far].mean())
        word_norm_far = float(word_norm[far].mean())

        def half_life(curve):
            half = 0.5 * float(curve[0])
            for d in range(curve.shape[0]):
                if float(curve[d]) <= half:
                    return d + 1
            return None

        def crossover(a, b):
            for d in range(a.shape[0]):
                if float(a[d]) < float(b[d]):
                    return d + 1
            return None

        starts_higher = bool(num_near > word_near)
        ends_lower = bool(num_far < word_far)
        cross = crossover(num_curve, word_curve)
        confirmed = bool(starts_higher and ends_lower)

        out["result"] = {
            "offsets": offsets,
            "number_curve": [round(float(x), 8) for x in num_curve],
            "word_curve": [round(float(x), 8) for x in word_curve],
            "number_curve_normalized": [round(float(x), 6) for x in num_norm],
            "word_curve_normalized": [round(float(x), 6) for x in word_norm],
            "offset1_number": round(num_near, 8),
            "offset1_word": round(word_near, 8),
            "far_region_offsets": [HORIZON // 2 + 1, HORIZON],
            "far_mean_number": round(num_far, 8),
            "far_mean_word": round(word_far, 8),
            "far_mean_number_normalized": round(num_norm_far, 6),
            "far_mean_word_normalized": round(word_norm_far, 6),
            "half_life_number": half_life(num_curve),
            "half_life_word": half_life(word_curve),
            "crossover_distance_number_below_word": cross,
            "starts_higher": starts_higher,
            "ends_lower": ends_lower,
            "faster_decay_normalized": bool(num_norm_far < word_norm_far),
            "prediction_confirmed": confirmed,
        }

        by_block = {}
        for r in records:
            by_block.setdefault(r["block"], {"number": [], "word_initial": []})[r["role"]].append(r["curve"])
        block_num_lower = []
        for b, d in by_block.items():
            if d["number"] and d["word_initial"]:
                nm = float(torch.tensor(d["number"]).mean(0)[far].mean())
                wm = float(torch.tensor(d["word_initial"]).mean(0)[far].mean())
                block_num_lower.append(nm < wm)
        if block_num_lower:
            out["result"]["n_blocks_paired"] = len(block_num_lower)
            out["result"]["far_frac_blocks_number_lower"] = round(sum(block_num_lower) / len(block_num_lower), 4)

    config.out_dir.mkdir(parents=True, exist_ok=True)
    with open(config.out_dir / "decay_profile.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    print("\n==== PERFIL DE DECAIMIENTO POR DISTANCIA ====")
    print("  validez (recon vs nativo, max abs):", round(validity_max_abs, 7))
    if num_curve is not None and word_curve is not None:
        r = out["result"]
        print(f"  n number {n_num} | n word {n_word} | horizon {HORIZON}")
        print(f"  pos media: number {out['position_mean_number']} | word {out['position_mean_word_initial']}")
        print(f"  offset 1:  number {r['offset1_number']:.3e} | word {r['offset1_word']:.3e}  (starts_higher={r['starts_higher']})")
        print(f"  far mean:  number {r['far_mean_number']:.3e} | word {r['far_mean_word']:.3e}  (ends_lower={r['ends_lower']})")
        print(f"  half-life: number {r['half_life_number']} | word {r['half_life_word']}")
        print(f"  cruce (number<word) en offset: {r['crossover_distance_number_below_word']}")
        if "far_frac_blocks_number_lower" in r:
            print(f"  pareado por bloque: {r['n_blocks_paired']} bloques | frac number<word (far): {r['far_frac_blocks_number_lower']}")
        print(f"  >>> PREDICCION CONFIRMADA: {r['prediction_confirmed']}")
        print("\n  curva (offset: number / word):")
        for i, off in enumerate(r["offsets"]):
            print(f"    {off:3d}: {r['number_curve'][i]:.3e} / {r['word_curve'][i]:.3e}")


if __name__ == "__main__":
    main()
