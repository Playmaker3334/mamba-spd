import torch
import torch.nn.functional as F
from transformers.models.mamba.modeling_mamba import MambaMixer

_STATE = {"active": False, "layer": None, "mask": None}
_ORIG = None


def _knockout_slow_forward(mixer, *args, **kwargs):
    if args:
        input_states = args[0]
    else:
        input_states = kwargs["input_states"]
    attention_mask = kwargs.get("attention_mask", None)
    if attention_mask is None and len(args) >= 4:
        attention_mask = args[3]

    batch_size, seq_len, _ = input_states.shape
    dtype = input_states.dtype

    projected_states = mixer.in_proj(input_states).transpose(1, 2)
    hidden_states, gate = projected_states.chunk(2, dim=1)

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    hidden_states = mixer.act(mixer.conv1d(hidden_states)[..., :seq_len])

    if attention_mask is not None:
        hidden_states = hidden_states * attention_mask.unsqueeze(1)

    ssm_parameters = mixer.x_proj(hidden_states.transpose(1, 2))
    time_step, B, C = torch.split(
        ssm_parameters, [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size], dim=-1
    )
    discrete_time_step = mixer.dt_proj(time_step)
    discrete_time_step = F.softplus(discrete_time_step).transpose(1, 2)

    A = -torch.exp(mixer.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
    discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
    deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

    if _STATE["active"] and mixer.layer_idx == _STATE["layer"] and _STATE["mask"] is not None:
        m = _STATE["mask"][:, None, :, None].to(deltaB_u.dtype)
        deltaB_u = deltaB_u * (1.0 - m)

    ssm_state = torch.zeros(
        batch_size, mixer.intermediate_size, mixer.ssm_state_size,
        device=hidden_states.device, dtype=deltaB_u.dtype
    )
    scan_outputs = []
    for i in range(seq_len):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        scan_outputs.append(torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))[:, :, 0])
    scan_output = torch.stack(scan_outputs, dim=-1)

    scan_output = scan_output + (hidden_states * mixer.D[None, :, None])
    scan_output = scan_output * mixer.act(gate)

    return mixer.out_proj(scan_output.transpose(1, 2))


def install():
    global _ORIG
    if _ORIG is None:
        _ORIG = MambaMixer.slow_forward
    MambaMixer.slow_forward = _knockout_slow_forward


def uninstall():
    global _ORIG
    if _ORIG is not None:
        MambaMixer.slow_forward = _ORIG
        _ORIG = None


def set_knockout(layer, mask):
    _STATE["active"] = True
    _STATE["layer"] = layer
    _STATE["mask"] = mask


def clear_knockout():
    _STATE["active"] = False
    _STATE["mask"] = None


@torch.no_grad()
def logits_for(model, block, layer, mask=None):
    if mask is None:
        clear_knockout()
    else:
        set_knockout(layer, mask)
    out = model(block).logits.float()
    clear_knockout()
    return out