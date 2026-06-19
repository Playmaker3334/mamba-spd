
import subprocess
import sys
from pathlib import Path

REPO_DIR = None


def find_repo():
    if REPO_DIR is not None:
        return Path(REPO_DIR).resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here,
        here / "mamba-spd",
        Path("/kaggle/working/mamba-spd"),
        Path.cwd(),
        Path.cwd() / "mamba-spd",
    ]
    for c in candidates:
        if (c / "scripts").is_dir() and (c / "src").is_dir():
            return c.resolve()
    raise SystemExit(
        "No se encontró el repositorio (carpeta con scripts/ y src/). "
        "Definir REPO_DIR al inicio de pipeline.py."
    )


def ensure_dep(mod):
    import importlib.util
    if importlib.util.find_spec(mod) is None:
        print(f"instalando dependencia: {mod}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod])


def _read(p):
    return Path(p).read_text(encoding="utf-8")


def _write(p, s):
    Path(p).write_text(s, encoding="utf-8")


def patch_batch_attrs(path):
    old = "(config.batch_size, config.seq_len)"
    new = "(config.batch, config.max_len)"
    c = _read(path)
    if old in c:
        _write(path, c.replace(old, new))
        return "parcheado (batch/max_len)"
    return "ya correcto"


def patch_depth_layers(path):
    old = "depth_profile(loader, blocks, config, metrics.categorize_token)"
    new = ("depth_profile(loader, blocks, config, metrics.categorize_token, "
           "layers=list(range(loader.model.config.num_hidden_layers)))")
    c = _read(path)
    if old in c:
        _write(path, c.replace(old, new))
        return "parcheado (48 capas)"
    return "ya correcto"


_L0_FUNC = '''def l0_per_token(ci_dict, threshold=0.0):
    stacked = torch.cat([ci_dict[n] for n in ci_dict], dim=-1)
    return float((stacked > threshold).float().sum(-1).mean())'''


def patch_l0(path):
    c = _read(path)
    if "def l0_per_token" in c:
        return "ya correcto"
    c = c.replace("import torch\n", "import torch\n\n\n" + _L0_FUNC + "\n", 1)
    _write(path, c)
    return "parcheado (l0_per_token)"


def spd_available(repo):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        from src.config import Config
        spd_path = str(Config().spd_path)
    except Exception:
        spd_path = "/kaggle/working/spd"
    if spd_path and Path(spd_path).exists() and spd_path not in sys.path:
        sys.path.insert(0, spd_path)
    import importlib.util
    try:
        return importlib.util.find_spec("spd") is not None
    except Exception:
        return False


def run_script(repo, name):
    print("\n" + "=" * 70, flush=True)
    print(f">>> {name}", flush=True)
    print("=" * 70, flush=True)
    r = subprocess.run([sys.executable, "-u", str(Path("scripts") / name)], cwd=str(repo))
    return r.returncode


def main():
    repo = find_repo()
    print(f"repositorio: {repo}", flush=True)
    ensure_dep("datasets")

    print("\n--- aplicando correcciones ---", flush=True)
    print(f"  src/metrics.py             : {patch_l0(repo / 'src' / 'metrics.py')}", flush=True)
    print(f"  scripts/01_static_decomp.py: {patch_batch_attrs(repo / 'scripts' / '01_static_decomp.py')}", flush=True)
    print(f"  scripts/03_intervention.py : {patch_batch_attrs(repo / 'scripts' / '03_intervention.py')}", flush=True)
    print(f"  scripts/07_depth_profile.py: {patch_depth_layers(repo / 'scripts' / '07_depth_profile.py')}", flush=True)

    dynamic = [
        "02_dynamic_decomp.py",
        "06_delta_forgetting.py",
        "07_depth_profile.py",
        "05_dynamic_ablation.py",
    ]
    static = [
        "01_static_decomp.py",
        "03_intervention.py",
        "04_generation_probe.py",
    ]

    results = {}

    print("\n--- pipeline dinámico ---", flush=True)
    for s in dynamic:
        results[s] = "OK" if run_script(repo, s) == 0 else "FALLO"

    if spd_available(repo):
        print("\n--- pipeline estático (spd disponible) ---", flush=True)
        for s in static:
            results[s] = "OK" if run_script(repo, s) == 0 else "FALLO"
    else:
        print("\nspd no disponible en config.spd_path: se omite el pipeline estático (01, 03, 04)", flush=True)
        for s in static:
            results[s] = "OMITIDO (sin spd)"

    print("\n" + "=" * 70, flush=True)
    print("RESUMEN", flush=True)
    print("=" * 70, flush=True)
    for s in dynamic + static:
        print(f"  {s:28} {results[s]}", flush=True)


if __name__ == "__main__":
    main()