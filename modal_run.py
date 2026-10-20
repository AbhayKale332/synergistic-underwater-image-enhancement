"""Pull the public GitHub source and run FUGN on the existing underwater volume.

    modal run --env gpu --detach modal_run.py --commit <git-sha>
    modal volume get --env gpu underwater results/pilot ./results/pilot

Data preparation runs on CPU before the A10G is allocated. All notebook code
comes from the requested public commit, rather than a copy of local changes.
"""

import json
import os
from pathlib import Path
import subprocess

import modal

REPOSITORY = "https://github.com/AbhayKale332/synergistic-underwater-image-enhancement.git"
MOUNT = "/mnt/underwater"
app = modal.App("fugn-uieb-pilot")
volume = modal.Volume.from_name("underwater")
data_image = modal.Image.debian_slim(python_version="3.12").apt_install("git", "unar").pip_install(
    "gdown>=5.2,<6", "Pillow>=10,<13", "nbformat>=5.10,<6", "rarfile==4.2"
)
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libglib2.0-0", "unar")
    .pip_install_from_requirements("requirements.txt")
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
)


def checkout(commit):
    if not commit or any(c not in "0123456789abcdef" for c in commit) or len(commit) != 40:
        raise ValueError("Pass a full 40-character Git commit SHA.")
    root = Path(MOUNT) / "source" / commit
    root.parent.mkdir(parents=True, exist_ok=True)
    if not (root / ".git").exists():
        subprocess.run(["git", "clone", REPOSITORY, str(root)], check=True)
    subprocess.run(["git", "fetch", "origin", commit], cwd=root, check=True)
    subprocess.run(["git", "checkout", "--detach", commit], cwd=root, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != commit:
        raise RuntimeError("Git checkout does not match the requested commit.")
    return root


@app.function(image=data_image, volumes={MOUNT: volume}, timeout=7200, cpu=4, memory=8192)
def prepare(commit: str):
    import nbformat

    root = checkout(commit)
    os.environ["FUGN_DATA_DIR"] = f"{MOUNT}/data/UIEB"
    os.environ["FUGN_OUTPUT_DIR"] = f"{MOUNT}/results/pilot"
    notebook = nbformat.read(root / "FUGN_UIEB.ipynb", as_version=4)
    scope = {"__name__": "fugn_data_preparation"}
    for cell in notebook.cells:
        if "data-definitions" in cell.metadata.get("tags", []):
            exec(compile(cell.source, "FUGN_UIEB.ipynb", "exec"), scope)
    result = scope["prepare_data"](scope["make_config"]())
    volume.commit()
    return {key: len(result[key]) for key in ("train", "val", "test", "challenging")}


@app.function(
    image=gpu_image, gpu="A10G", volumes={MOUNT: volume}, timeout=86400,
    cpu=8, memory=32768,
)
def train(commit: str, epochs: int = 10, preset: str = "pilot", run_name: str = "pilot",
          resume: bool = True):
    import nbformat
    from nbclient import NotebookClient
    import torch
    import shutil
    import tempfile

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the remote experiment.")
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("run_name may contain only letters, digits, underscores and hyphens.")
    volume.reload()
    root = checkout(commit)
    output = Path(MOUNT) / "results" / run_name
    output.mkdir(parents=True, exist_ok=True)
    # Large sequential archive reads avoid thousands of cold per-file volume
    # fetches. The canonical data remains on the attached persistent volume.
    persistent_data = Path(MOUNT) / "data" / "UIEB"
    working_data = Path(tempfile.mkdtemp(prefix="fugn-uieb-"))
    for subset in ("raw", "reference", "challenging"):
        print(f"Staging {subset} archive to local scratch ...", flush=True)
        shutil.copyfile(persistent_data / f"{subset}.archive", working_data / f"{subset}.archive")
    for manifest in persistent_data.glob("split_seed_*.json"):
        shutil.copyfile(manifest, working_data / manifest.name)
    os.environ.update({
        "FUGN_DATA_DIR": str(working_data), "FUGN_OUTPUT_DIR": str(output),
        "FUGN_PERSISTENT_DATA_DIR": str(persistent_data),
        "FUGN_EPOCHS": str(epochs), "FUGN_PRESET": preset,
        "FUGN_RESUME": "1" if resume else "0", "FUGN_COMMIT": commit,
        "FUGN_REQUIRE_CUDA": "1",
    })
    print(f"GPU: {torch.cuda.get_device_name(0)}; source: {commit}", flush=True)
    notebook = nbformat.read(root / "FUGN_UIEB.ipynb", as_version=4)
    executed_path = output / "FUGN_UIEB_executed.ipynb"

    async def persist_cell(**kwargs):
        nbformat.write(notebook, executed_path)
        # Checkpoints and results survive a disconnect or a failed later cell.
        await volume.commit.aio()

    client = NotebookClient(notebook, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(root)}},
                            on_cell_executed=persist_cell)
    try:
        client.execute()
    finally:
        nbformat.write(notebook, executed_path)
        volume.commit()
    manifest = json.loads((output / "summary.json").read_text())
    print(json.dumps(manifest, indent=2), flush=True)
    return str(output / "fugn_uieb_pilot.html")


@app.local_entrypoint()
def main(commit: str, epochs: int = 10, preset: str = "pilot", run_name: str = "pilot",
         resume: bool = True, prepare_only: bool = False):
    print("Preparing UIEB on CPU:", prepare.remote(commit))
    if not prepare_only:
        print("HTML report:", train.remote(commit, epochs, preset, run_name, resume))
