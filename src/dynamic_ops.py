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
    def __init__(self, d_op, n_atoms):
        super().__init__()
        self.encoder = nn.Linear(d_op, n_atoms)
        self.decoder = nn.Linear(n_atoms, d_op, bias=False)

    def forward(self, o):
        code = torch.relu(self.encoder(o))
        recon = self.decoder(code)
        return recon, code


class DynamicOperatorDictionary:
    def __init__(self, config):
        self.config = config
        self.dictionary = None
        self.d_op = None

    def collect_operators(self, loader, corpus, layer_idx):
        with OperatorExtractor(loader.model, layers=[layer_idx]) as ext:
            with torch.no_grad():
                for batch in corpus:
                    loader.model(batch.to(self.config.device))
        return build_operator_matrix(ext.operators, layer_idx)

    def fit(self, operators):
        self.d_op = operators.shape[-1]
        self.dictionary = OperatorDictionary(self.d_op, self.config.n_dynamic_atoms).to(operators.device)
        opt = optim.AdamW(self.dictionary.parameters(), lr=self.config.lr)
        history = []
        for step in range(self.config.n_steps):
            recon, code = self.dictionary(operators)
            mse = (recon - operators).pow(2).mean()
            sparsity = code.abs().mean()
            loss = mse + self.config.sparsity_coeff * sparsity
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                l0 = (code > 1e-4).float().sum(-1).mean()
            history.append({"step": step, "loss": float(loss), "mse": float(mse), "l0": float(l0)})
        return history

    def atom_usage(self, operators):
        with torch.no_grad():
            _, code = self.dictionary(operators)
        active = (code > 1e-4).float()
        return {
            "mean_l0": float(active.sum(-1).mean()),
            "atom_frequency": [round(x, 4) for x in active.mean(0).tolist()],
        }