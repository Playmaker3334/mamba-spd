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

    # corpus de entrenamiento: solo referencia de frecuencia (consistente con 06)
    train_blocks = build_corpus_blocks(loader, config)
    # held-out: el knockout se evalua sobre validacion, no sobre train
    eval_blocks = build_eval_blocks(loader, config, n_blocks=200)
    n_eval = eval_blocks.shape[0]
    L = eval_blocks.shape[1]

    # vocabulario completo: eval (validacion) puede contener ids > max(train)
    V = model.config.vocab_size
    raws = tok.convert_ids_to_tokens(list(range(V)))
    cat_of_id = [categorize_token(r) for r in raws]
    counts = torch.bincount(train_blocks.reshape(-1), minlength=V).float()

    ik.install()

    sample = eval_blocks[:1].to(config.device)
    with torch.no_grad():
        native = model(sample).logits.float()
    recon = ik.logits_for(model, sample, layer, mask=None)
    validity_max_abs = float((recon - native).abs().max())

    max_pairs_per_block = 3
    min_gap = 4

    records = []
    for bi in range(n_eval):
        block = eval_blocks[bi:bi + 1].to(config.device)
        ids = eval_blocks[bi].tolist()
        cats = [cat_of_id[t] for t in ids]
        num_pos = [p for p in range(L - min_gap) if cats[p] == "number"]
        word_pos = [p for p in range(L - min_gap) if cats[p] == "word_initial"]
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
                fut = kl[pos + 1:]
                records.append({
                    "block": bi,
                    "role": role,
                    "position": pos,
                    "token_id": ids[pos],
                    "logfreq": round(float(torch.log(counts[ids[pos]] + 1e-9)), 4),
                    "n_future": int(fut.shape[0]),
                    "kl_next": round(float(kl[pos + 1]), 8),
                    "kl_future_mean": round(float(fut.mean()), 8),
                    "kl_future_sum": round(float(fut.sum()), 8),
                })

    ik.uninstall()

    def agg(role):
        rs = [r for r in records if r["role"] == role]
        if not rs:
            return {}
        t = lambda k: torch.tensor([r[k] for r in rs])
        return {
            "n": len(rs),
            "kl_next_mean": round(float(t("kl_next").mean()), 8),
            "kl_next_median": round(float(t("kl_next").median()), 8),
            "kl_future_mean_mean": round(float(t("kl_future_mean").mean()), 8),
            "kl_future_mean_median": round(float(t("kl_future_mean").median()), 8),
            "position_mean": round(float(t("position").float().mean()), 2),
            "logfreq_mean": round(float(t("logfreq").mean()), 4),
        }

    paired = {}
    by_block = {}
    for r in records:
        by_block.setdefault(r["block"], {"number": [], "word_initial": []})[r["role"]].append(r)
    diffs_next, diffs_future = [], []
    for b, d in by_block.items():
        if d["number"] and d["word_initial"]:
            nn_ = torch.tensor([x["kl_future_mean"] for x in d["number"]]).mean()
            ww = torch.tensor([x["kl_future_mean"] for x in d["word_initial"]]).mean()
            diffs_future.append(float(nn_ - ww))
            nn2 = torch.tensor([x["kl_next"] for x in d["number"]]).mean()
            ww2 = torch.tensor([x["kl_next"] for x in d["word_initial"]]).mean()
            diffs_next.append(float(nn2 - ww2))
    if diffs_future:
        df = torch.tensor(diffs_future)
        dn = torch.tensor(diffs_next)
        paired = {
            "n_blocks_paired": len(diffs_future),
            "future_mean_diff_number_minus_word": round(float(df.mean()), 8),
            "future_frac_blocks_number_lower": round(float((df < 0).float().mean()), 4),
            "next_mean_diff_number_minus_word": round(float(dn.mean()), 8),
            "next_frac_blocks_number_lower": round(float((dn < 0).float().mean()), 4),
        }

    out = {
        "design": "implicit_attention_knockout_dBu_zero",
        "prediction": "number contribution to future positions is lower than matched content-words (number knockout effect < word knockout effect)",
        "eval_split": "wikitext-2-raw-v1 validation (held-out); frequency reference from train split",
        "primary_metric": "kl_future_mean (aggregate over all future positions); kl_next is robustness cross-check",
        "pairing": "matched by position only (nearest word_initial); frequency NOT paired at pair-level, justified by decoupling in 06 (corr logfreq-delta = -0.014); logfreq_mean reported per role for transparency",
        "confirm_if": "by_role.number.kl_future_mean_mean < by_role.word_initial.kl_future_mean_mean AND paired_by_block.future_frac_blocks_number_lower > 0.5",
        "validity_note": "check position_mean balance across roles; if imbalanced, kl_next (fixed offset) is the clean read instead of kl_future_mean",
        "layer": layer,
        "n_layers": n_layers,
        "n_eval_blocks": n_eval,
        "seq_len": L,
        "validity_check_logits_max_abs_recon_minus_native": round(validity_max_abs, 7),
        "by_role": {"number": agg("number"), "word_initial": agg("word_initial")},
        "paired_by_block": paired,
        "n_records": len(records),
    }

    config.out_dir.mkdir(parents=True, exist_ok=True)
    with open(config.out_dir / "implicit_knockout.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

    print("\n==== KNOCKOUT POR CATEGORIA (datos crudos) ====")
    print("  chequeo validez (logits recon vs nativo, max abs):", round(validity_max_abs, 7))
    for role in ["number", "word_initial"]:
        a = out["by_role"][role]
        if a:
            print(f"  {role:14s} | n {a['n']:4d} | kl_next {a['kl_next_mean']:.6e} | kl_fut_mean {a['kl_future_mean_mean']:.6e} | pos {a['position_mean']:.1f} | logf {a['logfreq_mean']:.2f}")
    if paired:
        print("  pareado por bloque:")
        print(f"    n bloques {paired['n_blocks_paired']}")
        print(f"    dif futura (num - word): {paired['future_mean_diff_number_minus_word']:.6e} | frac bloques num<word: {paired['future_frac_blocks_number_lower']}")
        print(f"    dif next  (num - word): {paired['next_mean_diff_number_minus_word']:.6e} | frac bloques num<word: {paired['next_frac_blocks_number_lower']}")


if __name__ == "__main__":
    main()