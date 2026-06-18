from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    model_id: str = "state-spaces/mamba-790m-hf"
    device: str = "cuda"
    dtype: str = "float16"
    layer: int = None

    target_tokens: int = 250000
    max_len: int = 96
    batch: int = 24

    n_dynamic_atoms: int = 96
    topk: int = 8
    n_steps: int = 5000
    lr: float = 2e-3
    block_balance: bool = False

    min_active: int = 30
    lift_min_count: int = 5
    lift_min_global: int = 20

    spd_path: str = "/kaggle/working/spd"
    target_patterns: tuple = (
        "backbone.layers.*.mixer.x_proj",
        "backbone.layers.*.mixer.dt_proj",
    )
    n_components: int = 40
    recon_coeff: float = 1.0
    sparsity_coeff: float = 1e-2
    pnorm: float = 1.0

    out_dir: Path = Path("/kaggle/working/mamba_spd_results")

    @property
    def metrics_dir(self) -> Path:
        return self.out_dir