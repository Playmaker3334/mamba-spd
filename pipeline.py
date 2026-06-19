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
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod])


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
    print(f"\n>>> {name}", flush=True)
    r = subprocess.run([sys.executable, "-u", str(Path("scripts") / name)], cwd=str(repo))
    return r.returncode


def main():
    repo = find_repo()
    ensure_dep("datasets")

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

    for s in dynamic:
        if run_script(repo, s) != 0:
            print(f"FALLO: {s}", flush=True)

    if spd_available(repo):
        for s in static:
            if run_script(repo, s) != 0:
                print(f"FALLO: {s}", flush=True)
    else:
        print("\nspd no disponible: se omiten 01, 03, 04", flush=True)


if __name__ == "__main__":
    main()