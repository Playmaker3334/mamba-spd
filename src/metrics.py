import torch


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