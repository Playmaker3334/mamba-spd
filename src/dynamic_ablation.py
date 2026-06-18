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

    def _hook(self, module, inp, out):
        if self.state["mode"] == "clean":
            return None
        x = (out.float() - self.mean) / self.std
        x = x - self.sae.pre_bias
        acts = torch.relu(self.sae.encoder(x))
        v, i = acts.topk(self.sae.k, dim=-1)
        code = torch.zeros_like(acts).scatter_(-1, i, v)
        if self.state["mode"] == "ablate" and self.state["atom"] is not None:
            code[..., self.state["atom"]] = 0
        recon = self.sae.decoder(code) + self.sae.pre_bias
        return (recon * self.std + self.mean).to(out.dtype)

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
        x = ((desc - self.decomp.mean) / self.decomp.std).to(self.decomp.config.device)
        x = x - self.sae.pre_bias
        acts = torch.relu(self.sae.encoder(x))
        v, i = acts.topk(self.sae.k, dim=-1)
        code = torch.zeros_like(acts).scatter_(-1, i, v)
        return (code[:, atom] > 1e-6).cpu()

    def run(self, eval_blocks, atom):
        handle = self.xproj.register_forward_hook(self._hook)
        nb, sl = eval_blocks.shape
        kl_recon = torch.zeros(nb, sl)
        kl_ablate = torch.zeros(nb, sl)
        nll_clean = torch.zeros(nb, sl - 1)
        nll_recon = torch.zeros(nb, sl - 1)
        nll_ablate = torch.zeros(nb, sl - 1)
        bs = self.decomp.config.batch
        for s in range(0, nb, bs):
            batch = eval_blocks[s:s + bs].to(self.decomp.config.device)
            lc = self._logits(batch, "clean")
            lr = self._logits(batch, "recon")
            la = self._logits(batch, "ablate", atom)
            pc = F.log_softmax(lc, -1)
            pr = F.log_softmax(lr, -1)
            pa = F.log_softmax(la, -1)
            kl_recon[s:s + batch.shape[0]] = (pc.exp() * (pc - pr)).sum(-1).cpu()
            kl_ablate[s:s + batch.shape[0]] = (pr.exp() * (pr - pa)).sum(-1).cpu()
            tgt = batch[:, 1:]
            nll_clean[s:s + batch.shape[0]] = -pc[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
            nll_recon[s:s + batch.shape[0]] = -pr[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
            nll_ablate[s:s + batch.shape[0]] = -pa[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu()
        handle.remove()
        self.state["mode"] = "clean"

        mask = self._atom_mask(eval_blocks, atom).reshape(nb, sl)
        effect = kl_ablate
        on = mask
        off = ~mask
        ppl_recon = float(nll_recon.mean().exp())
        ppl_ablate = float(nll_ablate.mean().exp())
        delta_nll = (nll_ablate - nll_recon)
        on_pos = mask[:, :-1]
        off_pos = ~mask[:, :-1]
        on_eff = float(effect[on].mean()) if int(on.sum()) > 0 else 0.0
        off_eff = float(effect[off].mean()) if int(off.sum()) > 0 else 0.0
        return {
            "atom": int(atom),
            "eval_tokens": int(nb * sl),
            "global": {
                "kl_sae_recon_mean": round(float(kl_recon.mean()), 6),
                "kl_atom_effect_mean": round(float(effect.mean()), 6),
                "ppl_recon_baseline": round(ppl_recon, 4),
                "ppl_ablated": round(ppl_ablate, 4),
                "delta_nll_mean": round(float(delta_nll.mean()), 6),
            },
            "targeted": {
                "n_atom_active": int(on.sum()),
                "n_atom_inactive": int(off.sum()),
                "kl_effect_on_atom_tokens": round(on_eff, 6),
                "kl_effect_off_atom_tokens": round(off_eff, 6),
                "specificity_ratio": round(on_eff / (off_eff + 1e-9), 3),
                "delta_nll_on_atom_tokens": round(float(delta_nll[on_pos].mean()) if int(on_pos.sum()) > 0 else 0.0, 6),
                "delta_nll_off_atom_tokens": round(float(delta_nll[off_pos].mean()) if int(off_pos.sum()) > 0 else 0.0, 6),
            },
        }