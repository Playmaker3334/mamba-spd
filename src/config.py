from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    model_id: str = "state-spaces/mamba-130m-hf"
    device: str = "cuda"
    dtype: str = "float32"
    spd_path: str = "/kaggle/working/spd"

    target_patterns: tuple = (
        "backbone.layers.*.mixer.x_proj",
        "backbone.layers.*.mixer.dt_proj",
    )
    n_components: int = 40

    n_steps: int = 400
    lr: float = 1e-3
    batch_size: int = 4
    seq_len: int = 64
    seed: int = 0

    recon_coeff: float = 1.0
    sparsity_coeff: float = 1e-2
    pnorm: float = 1.0

    n_dynamic_atoms: int = 32
    corpus_tokens: int = 50000

    out_dir: Path = Path("outputs")

    @property
    def metrics_dir(self) -> Path:
        return self.out_dir / "metrics"

    @property
    def figures_dir(self) -> Path:
        return self.out_dir / "figures"
