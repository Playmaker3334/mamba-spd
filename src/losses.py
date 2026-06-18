import torch


def reconstruction_loss(target, predicted):
    return (target - predicted).pow(2).mean()


def sparsity_loss(activations, pnorm=1.0):
    return activations.abs().pow(pnorm).sum(-1).mean()


def total_loss(target, predicted, activations, recon_coeff, sparsity_coeff, pnorm=1.0):
    rec = reconstruction_loss(target, predicted)
    sp = sparsity_loss(activations, pnorm)
    loss = recon_coeff * rec + sparsity_coeff * sp
    return loss, {"recon": float(rec), "sparsity": float(sp)}
