import torch


def l0_per_token(causal_importances, threshold=0.1):
    out = {}
    for name, ci in causal_importances.items():
        out[name] = float((ci > threshold).float().sum(-1).mean())
    return out


def reconstruction_fidelity(target_out, reconstructed_out):
    diff = (reconstructed_out - target_out).abs().mean()
    scale = target_out.abs().mean() + 1e-8
    return float(1.0 - diff / scale)


def activation_consistency(ci_a, ci_b, threshold=0.1):
    a = (ci_a > threshold).float()
    b = (ci_b > threshold).float()
    inter = (a * b).sum(-1)
    union = ((a + b) > 0).float().sum(-1) + 1e-8
    return float((inter / union).mean())
