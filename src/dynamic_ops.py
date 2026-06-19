import torch
import torch.nn as nn
import torch.optim as optim
from transformers.models.mamba.modeling_mamba import MambaMixer


class OperatorExtractor:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.store = {"ts": [], "B": [], "C": [], "dmag": []}
        self.n_calls = 0

    def __enter__(self):
        self._orig = MambaMixer.slow_forward
        store = self.store
        layer = self.layer
        orig = self._orig
        ext = self
        self.n_calls = 0

        def patched(mixer, *args, **kwargs):
            if mixer.layer_idx == layer:
                ext.n_calls += 1
                x = args[0] if args else kwargs["input_states"]
                _, seq_len, _ = x.shape
                proj = mixer.in_proj(x).transpose(1, 2)
                hs, gate = proj.chunk(2, dim=1)
                hs = mixer.act(mixer.conv1d(hs)[..., :seq_len])
                ssm = mixer.x_proj(hs.transpose(1, 2))
                ts, B, C = torch.split(ssm, [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size], dim=-1)
                dmag = nn.functional.softplus(mixer.dt_proj(ts).float()).mean(-1)
                store["ts"].append(ts.float().detach().cpu())
                store["B"].append(B.float().detach().cpu())
                store["C"].append(C.float().detach().cpu())
                store["dmag"].append(dmag.detach().cpu())
            return orig(mixer, *args, **kwargs)

        MambaMixer.slow_forward = patched
        return self

    def __exit__(self, *a):
        MambaMixer.slow_forward = self._orig


def build_corpus_blocks(loader, config):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = []
    for row in ds:
        t = row["text"].strip()
        if len(t) < 20:
            continue
        ids.extend(loader.tokenizer(t, add_special_tokens=False).input_ids)
        if len(ids) >= config.target_tokens + config.max_len:
            break
    nb = config.target_tokens // config.max_len
    return torch.tensor(ids[:nb * config.max_len]).reshape(nb, config.max_len)


class OperatorDictionary(nn.Module):
    def __init__(self, d_op, n_atoms, k):
        super().__init__()
        self.k = k
        self.pre_bias = nn.Parameter(torch.zeros(d_op))
        self.encoder = nn.Linear(d_op, n_atoms)
        self.decoder = nn.Linear(n_atoms, d_op, bias=False)

    def forward(self, o):
        x = o - self.pre_bias
        acts = torch.relu(self.encoder(x))
        v, i = acts.topk(self.k, dim=-1)
        code = torch.zeros_like(acts).scatter_(-1, i, v)
        return self.decoder(code) + self.pre_bias, code


class DynamicOperatorDictionary:
    def __init__(self, config):
        self.config = config
        self.dictionary = None
        self.mean = None
        self.std = None
        self.descriptor = None
        self.dmag = None
        self.tokids = None
        self.positions = None
        self.code = None
        self.active = None
        self.fve = None
        self.layer = None
        self.block_dims = None

    def collect(self, loader, blocks):
        layer = self.config.layer if self.config.layer is not None else loader.model.config.num_hidden_layers // 2
        with OperatorExtractor(loader.model, layer) as ext:
            with torch.no_grad():
                for bi, i in enumerate(range(0, blocks.shape[0], self.config.batch)):
                    loader.model(blocks[i:i + self.config.batch].to(self.config.device))
                    if bi == 0 and ext.n_calls == 0:
                        raise RuntimeError(
                            "OperatorExtractor no capturó ningún operador: "
                            "MambaMixer.slow_forward no fue invocado. Es probable que los "
                            "kernels rápidos (mamba_ssm / causal_conv1d) estén instalados y "
                            "que el forward se enrute por cuda_kernels_forward. "
                            "Desinstálalos para forzar el camino lento."
                        )
        ts = torch.cat([x.reshape(-1, x.shape[-1]) for x in ext.store["ts"]], 0)
        B = torch.cat([x.reshape(-1, x.shape[-1]) for x in ext.store["B"]], 0)
        C = torch.cat([x.reshape(-1, x.shape[-1]) for x in ext.store["C"]], 0)
        self.dmag = torch.cat([x.reshape(-1) for x in ext.store["dmag"]], 0)
        self.descriptor = torch.cat([ts, B, C], -1)
        self.block_dims = (ts.shape[1], B.shape[1], C.shape[1])
        self.tokids = blocks.reshape(-1)
        self.positions = torch.arange(self.config.max_len).repeat(blocks.shape[0]).float() / self.config.max_len
        self.layer = layer
        return self.descriptor

    def fit(self):
        self.mean = self.descriptor.mean(0, keepdim=True)
        self.std = self.descriptor.std(0, keepdim=True) + 1e-6
        x = ((self.descriptor - self.mean) / self.std).to(self.config.device)
        d_op = x.shape[1]
        if self.config.block_balance:
            w = torch.empty(d_op)
            nb = len(self.block_dims)
            start = 0
            for bd in self.block_dims:
                w[start:start + bd] = (1.0 / nb) / bd
                start += bd
        else:
            w = torch.full((d_op,), 1.0 / d_op)
        w = w.to(self.config.device)
        self.dictionary = OperatorDictionary(d_op, self.config.n_dynamic_atoms, self.config.topk).to(self.config.device)
        opt = optim.AdamW(self.dictionary.parameters(), lr=self.config.lr)
        history = []
        for step in range(self.config.n_steps):
            r, code = self.dictionary(x)
            mse = (w * (r - x).pow(2)).sum(-1).mean()
            opt.zero_grad()
            mse.backward()
            opt.step()
            history.append({"step": step, "mse": float(mse.detach())})
        with torch.no_grad():
            r, code = self.dictionary(x)
        self.fve = 1 - float((w * (r - x).pow(2)).sum(-1).mean())
        self.code = code.cpu()
        self.active = self.code > 1e-6
        return history