import sys

import torch
import torch.optim as optim

from .losses import total_loss


class StaticDecomposer:
    def __init__(self, loader, config):
        if config.spd_path not in sys.path:
            sys.path.insert(0, config.spd_path)
        from spd.models.component_model import ComponentModel, init_As_and_Bs_

        self.loader = loader
        self.config = config
        self.cm = ComponentModel(
            base_model=loader.model,
            target_module_patterns=list(config.target_patterns),
            C=config.n_components,
            n_ci_mlp_neurons=0,
            pretrained_model_output_attr="logits",
        )
        self.components = {
            k.removeprefix("components.").replace("-", "."): v
            for k, v in self.cm.components.items()
        }
        self.gates = {
            k.removeprefix("gates.").replace("-", "."): v
            for k, v in self.cm.gates.items()
        }
        self.cm.to(config.device)
        init_As_and_Bs_(model=self.cm, components=self.components)

    def causal_importances(self, batch):
        from spd.models.component_utils import calc_causal_importances

        target_out, pre_acts = self.cm.forward_with_pre_forward_cache_hooks(
            input_ids=batch, module_names=list(self.components.keys())
        )
        As = {n: self.components[n].A for n in self.components}
        ci, _ = calc_causal_importances(
            pre_weight_acts=pre_acts, As=As, gates=self.gates, detach_inputs=False
        )
        return target_out, ci

    def trainable_params(self):
        params = []
        for n in self.components:
            params += list(self.components[n].parameters())
            params += list(self.gates[n].parameters())
        return params

    def train(self, data_sampler):
        opt = optim.AdamW(self.trainable_params(), lr=self.config.lr)
        history = []
        for step in range(self.config.n_steps):
            batch = data_sampler()
            opt.zero_grad()
            target_out, ci = self.causal_importances(batch)
            masked = self.cm.forward_with_components(
                input_ids=batch, components=self.components, masks=ci
            )
            ci_stack = torch.cat([ci[n] for n in ci], dim=-1)
            loss, terms = total_loss(
                target_out, masked, ci_stack,
                self.config.recon_coeff, self.config.sparsity_coeff, self.config.pnorm,
            )
            loss.backward()
            opt.step()
            history.append({"step": step, "loss": float(loss), **terms})
        return history
