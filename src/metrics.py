import torch


def l0_per_token(ci_dict, threshold=0.0):
    stacked = torch.cat([ci_dict[n] for n in ci_dict], dim=-1)
    return float((stacked > threshold).float().sum(-1).mean())


def pca_baseline(descriptor):
    Xn = (descriptor - descriptor.mean(0)) / (descriptor.std(0) + 1e-6)
    S = torch.linalg.svdvals(Xn - Xn.mean(0))
    frac = (S ** 2) / (S ** 2).sum()
    cum = torch.cumsum(frac, 0)
    return {
        "var_explained_at_8": round(float(cum[7]), 4),
        "n_components_90": int((cum < 0.9).sum()) + 1,
        "n_components_95": int((cum < 0.95).sum()) + 1,
        "top10_eigen_fraction": [round(float(x), 4) for x in frac[:10].tolist()],
    }


def atom_semantics(decomp, tokenizer):
    config = decomp.config
    active = decomp.active
    tokids = decomp.tokids
    dmag = decomp.dmag
    positions = decomp.positions
    freq = active.float().mean(0)
    N = tokids.shape[0]
    V = int(tokids.max()) + 1
    base = torch.bincount(tokids, minlength=V).float() / N
    basec = torch.bincount(tokids, minlength=V)
    gm, gs = float(dmag.mean()), float(dmag.std())
    atoms = []
    spec_tok = 0
    spec_delta = 0
    for a in range(config.n_dynamic_atoms):
        m = active[:, a]
        na = int(m.sum())
        if na < config.min_active:
            atoms.append({"id": a, "frequency": round(float(freq[a]), 4), "n_active": na, "status": "rare"})
            continue
        z = (float(dmag[m].mean()) - gm) / (gs + 1e-9)
        cnt = torch.bincount(tokids[m], minlength=V).float()
        lift = (cnt / na) / (base + 1e-9)
        valid = (cnt >= config.lift_min_count) & (basec >= config.lift_min_global)
        lift = lift * valid
        top = lift.argsort(descending=True)[:8]
        tt = [{"token": tokenizer.decode([int(t)]).strip(), "lift": round(float(lift[t]), 2), "count": int(cnt[t])} for t in top if lift[t] > 0]
        maxlift = float(lift.max())
        if maxlift > 3:
            spec_tok += 1
        if abs(z) > 1:
            spec_delta += 1
        atoms.append({"id": a, "frequency": round(float(freq[a]), 4), "n_active": na, "delta_z": round(z, 3),
                      "delta_mean": round(float(dmag[m].mean()), 5), "mean_position": round(float(positions[m].mean()), 3),
                      "max_lift": round(maxlift, 2), "top_tokens": tt})
    summary = {"lexically_specialized_lift_gt3": spec_tok, "delta_regime_abs_z_gt1": spec_delta,
               "delta_global_mean": round(gm, 5), "delta_global_std": round(gs, 5)}
    return atoms, summary


def coactivation(decomp, threshold=0.1):
    active = decomp.active.float()
    Ac = active - active.mean(0)
    cov = Ac.T @ Ac
    nrm = cov.diag().sqrt()
    corr = cov / (nrm[:, None] * nrm[None, :] + 1e-9)
    corr.fill_diagonal_(0)
    out = {}
    for a in range(corr.shape[0]):
        t3 = corr[a].argsort(descending=True)[:3]
        out[str(a)] = [{"atom": int(j), "corr": round(float(corr[a, j]), 3)} for j in t3 if corr[a, j] > threshold]
    return out


def categorize_token(raw):
    import re
    word_initial = raw.startswith("Ġ")
    s = raw[1:] if word_initial else raw
    if s == "":
        return "space"
    if re.fullmatch(r"\d+", s):
        return "number"
    if re.search(r"\d", s) and re.search(r"[A-Za-z]", s):
        return "alnum"
    if re.fullmatch(r"[^\w\s]+", s):
        return "punct_symbol"
    if s.isalpha():
        return "word_initial" if word_initial else "subword_cont"
    return "other"


def _pearson(a, b):
    import torch
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


def delta_by_category(decomp, tokenizer, baseline="word_initial"):
    import torch
    dmag = decomp.dmag
    tokids = decomp.tokids
    N = tokids.shape[0]
    V = int(tokids.max()) + 1
    raws = tokenizer.convert_ids_to_tokens(list(range(V)))
    cat_of_id = [categorize_token(r) for r in raws]
    cat_list = sorted(set(cat_of_id))
    cat_idx = {c: i for i, c in enumerate(cat_list)}
    id_to_cat = torch.tensor([cat_idx[c] for c in cat_of_id])
    pos_cat = id_to_cat[tokids]
    counts = torch.bincount(tokids, minlength=V).float()
    logf = torch.log(counts[tokids] + 1e-9)

    gm, gs = float(dmag.mean()), float(dmag.std())
    base_mask = pos_cat == cat_idx[baseline] if baseline in cat_idx else None
    base_d = dmag[base_mask] if base_mask is not None and int(base_mask.sum()) > 0 else None

    rows = []
    for c in cat_list:
        m = pos_cat == cat_idx[c]
        n = int(m.sum())
        if n == 0:
            continue
        d_c = dmag[m]
        row = {
            "category": c,
            "n_tokens": n,
            "frac_corpus": round(n / N, 4),
            "delta_mean": round(float(d_c.mean()), 5),
            "delta_median": round(float(d_c.median()), 5),
            "delta_std": round(float(d_c.std()), 5),
            "z_vs_global": round((float(d_c.mean()) - gm) / (gs + 1e-9), 3),
        }
        if base_d is not None and c != baseline:
            pooled = (((d_c.std() ** 2) + (base_d.std() ** 2)) / 2).sqrt()
            row["cohen_d_vs_" + baseline] = round((float(d_c.mean()) - float(base_d.mean())) / (float(pooled) + 1e-9), 3)
        rows.append(row)
    rows.sort(key=lambda r: r["delta_mean"])

    summary = {
        "layer": decomp.layer,
        "n_tokens": N,
        "delta_global_mean": round(gm, 5),
        "delta_global_std": round(gs, 5),
        "corr_logfreq_delta": round(_pearson(logf, dmag), 4),
        "baseline_category": baseline,
        "categories": rows,
    }
    return summary