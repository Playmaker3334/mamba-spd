import torch
import torch.nn.functional as F
from transformers.models.mamba.modeling_mamba import MambaMixer


class MultiLayerDeltaExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = set(layers)
        self.store = {l: [] for l in layers}

    def __enter__(self):
        self._orig = MambaMixer.slow_forward
        store = self.store
        layers = self.layers
        orig = self._orig

        def patched(mixer, *args, **kwargs):
            if mixer.layer_idx in layers:
                x = args[0] if args else kwargs["input_states"]
                _, seq_len, _ = x.shape
                proj = mixer.in_proj(x).transpose(1, 2)
                hs, gate = proj.chunk(2, dim=1)
                hs = mixer.act(mixer.conv1d(hs)[..., :seq_len])
                ssm = mixer.x_proj(hs.transpose(1, 2))
                ts = ssm[..., :mixer.time_step_rank]
                dmag = F.softplus(mixer.dt_proj(ts).float()).mean(-1)
                store[mixer.layer_idx].append(dmag.detach().cpu())
            return orig(mixer, *args, **kwargs)

        MambaMixer.slow_forward = patched
        return self

    def __exit__(self, *a):
        MambaMixer.slow_forward = self._orig


def collect_depth_deltas(loader, blocks, layers, batch, device):
    nb = blocks.shape[0]
    n_batches = (nb + batch - 1) // batch
    print(f"  capturando Δ en capas {layers} | {nb} bloques en {n_batches} batches", flush=True)
    with MultiLayerDeltaExtractor(loader.model, layers) as ext:
        with torch.no_grad():
            for bi, i in enumerate(range(0, nb, batch)):
                loader.model(blocks[i:i + batch].to(device))
                if bi % 10 == 0 or bi == n_batches - 1:
                    print(f"    batch {bi + 1}/{n_batches}", flush=True)
    return {l: torch.cat(v).reshape(-1) for l, v in ext.store.items()}


def depth_profile(loader, blocks, config, categorize_fn, layers=None):
    n_layers = loader.model.config.num_hidden_layers
    if layers is None:
        step = max(1, n_layers // 7)
        layers = sorted(set(list(range(0, n_layers, step)) + [n_layers - 1]))
    print(f"perfil por profundidad | modelo de {n_layers} capas | muestreando {len(layers)} capas", flush=True)
    deltas = collect_depth_deltas(loader, blocks, layers, config.batch, config.device)
    tokids = blocks.reshape(-1)
    V = int(tokids.max()) + 1
    raws = loader.tokenizer.convert_ids_to_tokens(list(range(V)))
    cat_of_id = [categorize_fn(r) for r in raws]
    cat_list = sorted(set(cat_of_id))
    cat_idx = {c: i for i, c in enumerate(cat_list)}
    id_to_cat = torch.tensor([cat_idx[c] for c in cat_of_id])
    pos_cat = id_to_cat[tokids]

    target_cats = [c for c in ["number", "punct_symbol", "subword_cont", "word_initial"] if c in cat_idx]
    profile = {}
    for c in target_cats:
        m = pos_cat == cat_idx[c]
        profile[c] = {}
        for l in layers:
            d = deltas[l]
            dl = d[m[:d.shape[0]]] if d.shape[0] == m.shape[0] else d[m]
            profile[c][str(l)] = round(float(dl.mean()), 5)

    baseline = "word_initial"
    cohen_by_layer = {}
    if baseline in cat_idx:
        bmask = pos_cat == cat_idx[baseline]
        for c in target_cats:
            if c == baseline:
                continue
            cmask = pos_cat == cat_idx[c]
            cohen_by_layer[c] = {}
            for l in layers:
                d = deltas[l]
                dc = d[cmask]
                db = d[bmask]
                pooled = (((dc.std() ** 2) + (db.std() ** 2)) / 2).sqrt()
                cohen_by_layer[c][str(l)] = round((float(dc.mean()) - float(db.mean())) / (float(pooled) + 1e-9), 3)

    return {
        "n_layers": n_layers,
        "layers_sampled": layers,
        "n_tokens": int(tokids.shape[0]),
        "baseline_category": baseline,
        "delta_mean_by_category_by_layer": profile,
        "cohen_d_vs_baseline_by_layer": cohen_by_layer,
    }