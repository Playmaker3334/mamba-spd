import torch


class OperatorAblator:
    def __init__(self, decomposer):
        self.decomposer = decomposer
        self.cm = decomposer.cm

    @torch.no_grad()
    def ablate_component(self, batch, layer_name, component_idx):
        target_out, ci = self.decomposer.causal_importances(batch)
        baseline = self.cm.forward_with_components(
            input_ids=batch, components=self.decomposer.components, masks=ci
        )
        ablated = {n: c.clone() for n, c in ci.items()}
        ablated[layer_name][..., component_idx] = 0.0
        modified = self.cm.forward_with_components(
            input_ids=batch, components=self.decomposer.components, masks=ablated
        )
        delta = (modified - baseline).abs().mean()
        return float(delta)
