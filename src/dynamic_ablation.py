import torch
import torch.nn.functional as F
from src.dynamic_ops import OperatorExtractor


def select_target_atom(atoms, mode="max_delta_z"):
    candidates = [a for a in atoms if "delta_z" in a]
    if mode == "max_delta_z":
        target = max(candidates, key=lambda a: a["delta_z"])
    elif mode == "min_delta_z":
        target = min(candidates, key=lambda a: a["delta_z"])
    elif mode == "abs_delta_z":
        target = max(candidates, key=lambda a: abs(a["delta_z"]))
    else:
        target = max(candidates, key=lambda a: a["max_lift"])
    return target["id"], target


class AblationProbe:
    def __init__(self, decomp, loader):
        self.decomp = decomp
        self.loader = loader
        self.model = loader.model
        self.layer = decomp.layer
        self.xproj = self.model.backbone.layers[self.layer].mixer.x_proj
        self.mean = decomp.mean.to(decomp.config.device)
        self.std = decomp.std.to(decomp.config.device)
        self.sae = decomp.dictionary
        self.state = {"mode": "clean", "atom": None}
        self._contrib_norm = []

    def _hook(self, module, inp, out):
        if self.state["mode"] == "clean":
            return None
        raw = out.float()
        x = (raw - self.mean) / self.std - self.sae.pre_bias
        acts = torch.relu(self.sae.encoder(x))
        v, i = acts.topk(self.sae.k, dim=-1)
        code = torch.zeros_like(acts).scatter_(-1, i, v)
        atom = self.state["atom"]
        w = self.sae.decoder.weight[:, atom]
        contrib = code[..., atom:atom + 1] * w.unsqueeze(0).unsqueeze(0) * self.std
        self._contrib_norm.append(contrib.norm(dim=-1).detach().cpu().reshape(-1))
        return (raw - contrib).to(out.dtype)

    def _logits(self, batch, mode, atom=None):
        self.state["mode"] = mode
        self.state["atom"] = atom
        with torch.no_grad():
            return self.model(batch).logits.float()

    def _atom_mask(self, eval_blocks, atom):
        with OperatorExtractor(self.model, self.layer) as ext:
            with torch.no_grad():
                for i in range(0, eval_blocks.shape[0], self.decomp.config.batch):
                    self.model(eval_blocks[i:i + self.decomp.config.batch].to(self.decomp.config.device))
        ts = torch.cat([x.reshape(-1, x.shape[-1]) for x in ext.store["ts"]], 0)
        B = torch.cat([x.reshape(-1, x.shape[-1]) for x in ext.store["B"]], 0)
        C = torch.cat([x.reshape(-1, x.shape[-1]) for x in ext.store["C"]], 0)
        desc = torch.cat([ts, B, C], -1)
        x = ((desc - self.decomp.mean) / self.decomp.std).to(self.decomp.config.device) - self.sae.pre_bias
        acts = torch.relu(self.sae.encoder(x))
        v, i = acts.topk(self.sae.k, dim=-1)
        code = torch.zeros_like(acts).scatter_(-1, i, v)
        return (code[:, atom] > 1e-6).cpu()

    def run(self, eval_blocks, atom):
        handle = self.xproj.register_forward_hook(self._hook)
        nb, sl = eval_blocks.shape
        kl_effect = torch.zeros(nb, sl)
        nll_clean = torch.zeros(nb, sl - 1)
        nll_ablate = torch.zeros(nb, sl - 1)
        bs = self.decomp.config.batch
        self._contrib_norm = []
        for s in range(0, nb, bs):
            batch = eval_blocks[s:s + bs].to(self.decomp.config.device)
            lc = self._logits(batch, "clean")
            la = self._logits(batch, "ablate", atom)
            pc = F.log_softmax(lc, -1)
            pa = F.log_softmax(la, -1)
            kl_effect[s:s + batch.shape[0]] = (pc.exp() * (pc - pa)).sum(-1).cpu()
            tgt = batch[:, 1:]
            nll_clean[s:s + batch.shape[0]] = -pc[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
            nll_ablate[s:s + batch.shape[0]] = -pa[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
        handle.remove()
        self.state["mode"] = "clean"

        mask = self._atom_mask(eval_blocks, atom).reshape(nb, sl)
        on = mask
        off = ~mask
        next_on = torch.zeros_like(mask)
        next_on[:, 1:] = mask[:, :-1]
        next_on = next_on & ~mask

        delta_nll = nll_ablate - nll_clean
        on_pos = mask[:, :-1]
        off_pos = ~mask[:, :-1]
        ppl_clean = float(nll_clean.mean().exp())
        ppl_ablate = float(nll_ablate.mean().exp())
        contrib = torch.cat(self._contrib_norm) if self._contrib_norm else torch.zeros(1)
        contrib = contrib.reshape(nb, sl)

        on_eff = float(kl_effect[on].mean()) if int(on.sum()) > 0 else 0.0
        off_eff = float(kl_effect[off].mean()) if int(off.sum()) > 0 else 0.0
        next_eff = float(kl_effect[next_on].mean()) if int(next_on.sum()) > 0 else 0.0
        return {
            "atom": int(atom),
            "eval_tokens": int(nb * sl),
            "design": "surgical_atom_subtraction",
            "global": {
                "kl_effect_mean": round(float(kl_effect.mean()), 6),
                "ppl_clean": round(ppl_clean, 4),
                "ppl_ablated": round(ppl_ablate, 4),
                "ppl_ratio": round(ppl_ablate / ppl_clean, 4),
                "delta_nll_mean": round(float(delta_nll.mean()), 6),
            },
            "targeted": {
                "n_atom_active": int(on.sum()),
                "n_atom_inactive": int(off.sum()),
                "n_next_after_active": int(next_on.sum()),
                "kl_on_atom_tokens": round(on_eff, 6),
                "kl_off_atom_tokens": round(off_eff, 6),
                "kl_next_after_active": round(next_eff, 6),
                "specificity_ratio": round(on_eff / (off_eff + 1e-9), 3),
                "delta_nll_on_atom_tokens": round(float(delta_nll[on_pos].mean()) if int(on_pos.sum()) > 0 else 0.0, 6),
                "delta_nll_off_atom_tokens": round(float(delta_nll[off_pos].mean()) if int(off_pos.sum()) > 0 else 0.0, 6),
            },
            "intervention": {
                "contrib_norm_on_active_mean": round(float(contrib[on].mean()) if int(on.sum()) > 0 else 0.0, 5),
                "descriptor_dim": int(self.sae.decoder.weight.shape[0]),
            },
        }