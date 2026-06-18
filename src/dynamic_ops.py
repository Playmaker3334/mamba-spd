import torch
import torch.nn as nn
import torch.optim as optim
from transformers.models.mamba.modeling_mamba import MambaMixer


class OperatorExtractor:
    def __init__(self, model, layers=None):
        self.model = model
        self.layers = layers
        self.operators = {}

    def __enter__(self):
        self._orig = MambaMixer.slow_forward
        store = self.operators
        keep = self.layers
        orig = self._orig

        def patched(mixer, *args, **kwargs):
            input_states = args[0] if args else kwargs["input_states"]
            _, seq_len, _ = input_states.shape
            projected = mixer.in_proj(input_states).transpose(1, 2)
            hidden_states, gate = projected.chunk(2, dim=1)
            hidden_states = mixer.act(mixer.conv1d(hidden_states)[..., :seq_len])
            ssm_parameters = mixer.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters,
                [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size],
                dim=-1,
            )
            discrete_time_step = nn.functional.softplus(mixer.dt_proj(time_step))
            if keep is None or mixer.layer_idx in keep:
                rec = store.setdefault(mixer.layer_idx, {"delta": [], "B": [], "C": []})
                rec["delta"].append(discrete_time_step.detach().cpu())
                rec["B"].append(B.detach().cpu())
                rec["C"].append(C.detach().cpu())
            return orig(mixer, *args, **kwargs)

        MambaMixer.slow_forward = patched
        return self

    def __exit__(self, *a):
        MambaMixer.slow_forward = self._orig


def build_operator_matrix(store, layer_idx):
    rec = store[layer_idx]
    delta = torch.cat([d.reshape(-1, d.shape[-1]) for d in rec["delta"]], dim=0)
    B = torch.cat([b.reshape(-1, b.shape[-1]) for b in rec["B"]], dim=0)
    C = torch.cat([c.reshape(-1, c.shape[-1]) for c in rec["C"]], dim=0)
    return torch.cat([delta, B, C], dim=-1)


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
        topv, topi = acts.topk(self.k, dim=-1)
        code = torch.zeros_like(acts).scatter_(-1, topi, topv)
        recon = self.decoder(code) + self.pre_bias
        return recon, code


class DynamicOperatorDictionary:
    def __init__(self, config):
        self.config = config
        self.dictionary = None
        self.d_op = None
        self.mean = None
        self.std = None

    def _normalize(self, operators):
        return (operators - self.mean) / self.std

    def collect_operators(self, loader, corpus, layer_idx):
        with OperatorExtractor(loader.model, layers=[layer_idx]) as ext:
            with torch.no_grad():
                for batch in corpus:
                    loader.model(batch.to(self.config.device))
        return build_operator_matrix(ext.operators, layer_idx)

    def fit(self, operators):
        self.d_op = operators.shape[-1]
        self.mean = operators.mean(0, keepdim=True)
        self.std = operators.std(0, keepdim=True) + 1e-6
        x = self._normalize(operators)
        k = getattr(self.config, "topk", 8)
        self.dictionary = OperatorDictionary(self.d_op, self.config.n_dynamic_atoms, k).to(x.device)
        opt = optim.AdamW(self.dictionary.parameters(), lr=self.config.lr)
        history = []
        for step in range(self.config.n_steps):
            recon, code = self.dictionary(x)
            mse = (recon - x).pow(2).mean()
            opt.zero_grad()
            mse.backward()
            opt.step()
            with torch.no_grad():
                l0 = (code > 1e-6).float().sum(-1).mean()
            history.append({"step": step, "mse": float(mse.detach()), "l0": float(l0.detach())})
        return history

    def atom_usage(self, operators):
        x = self._normalize(operators)
        with torch.no_grad():
            recon, code = self.dictionary(x)
        active = (code > 1e-6).float()
        mse = float((recon - x).pow(2).mean())
        return {
            "mean_l0": float(active.sum(-1).mean()),
            "variance_explained": round(1.0 - mse, 4),
            "recon_mse_normalized": round(mse, 4),
            "dead_atoms": int(sum(1 for f in active.mean(0).tolist() if f < 0.01)),
            "atom_frequency": [round(v, 4) for v in active.mean(0).tolist()],
        }