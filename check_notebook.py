"""Small integration check for the notebook's actual code, including resume.

    python check_notebook.py
    python check_notebook.py --artifacts results/pilot

Synthetic images here are temporary test fixtures, never experiment data.
"""

import argparse
import json
from pathlib import Path
import tempfile


def load_definitions():
    notebook = json.loads(Path(__file__).with_name("FUGN_UIEB.ipynb").read_text())
    scope = {"__name__": "fugn_notebook_checks"}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, "FUGN_UIEB.ipynb", "exec")
        if {"definitions", "data-definitions"} & set(cell.get("metadata", {}).get("tags", [])):
            exec(source, scope)
    return scope


def check_artifacts(directory):
    import pandas as pd
    from html.parser import HTMLParser

    root = Path(directory)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert summary["paired_comparisons"] == 90 and summary["challenging_comparisons"] == 60
    table = pd.read_csv(root / "metrics.csv")
    assert len(table) == 300
    assert table.groupby(["split", "image"]).size().eq(2).all()
    assert table.loc[table.split == "challenging", ["PSNR", "SSIM", "FSIM"]].isna().all().all()
    assert table.loc[table.split == "test", ["PSNR", "SSIM", "FSIM"]].notna().all().all()
    assert len(list((root / "enhanced" / "test").glob("*.png"))) == 90
    assert len(list((root / "enhanced" / "challenging").glob("*.png"))) == 60
    assert len(list((root / "comparisons").glob("*.png"))) == 30

    class ReportParser(HTMLParser):
        images = 0

        def handle_starttag(self, tag, attributes):
            attrs = dict(attributes)
            if tag == "img":
                assert attrs["src"].startswith("data:image/png;base64,")
                assert attrs.get("alt")
                self.images += 1
            if tag in {"script", "iframe"}:
                raise AssertionError("The report must be standalone and script-free")

    parser = ReportParser()
    parser.feed((root / "fugn_uieb_pilot.html").read_text())
    assert parser.images == 36
    print("Artifact checks passed: 150 comparisons, 300 metric records, 36 embedded figures.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts")
    args = parser.parse_args()
    if args.artifacts:
        check_artifacts(args.artifacts)
        return
    scope = load_definitions()
    torch, np, Image = scope["torch"], scope["np"], scope["Image"]
    torch.set_num_threads(2)
    scope["run_self_checks"]("cpu", full_model=True)
    with tempfile.TemporaryDirectory(prefix="fugn-test-only-") as temporary:
        root = Path(temporary)
        records = []
        for index in range(4):
            path = root / f"TEST_ONLY_{index}.png"
            array = np.random.default_rng(index).integers(0, 256, (40, 40, 3), dtype=np.uint8)
            Image.fromarray(array).save(path)
            records.append((path, path, str(index)))
        splits = {"train": records[:2], "val": records[2:], "manifest": {"fingerprint": "TEST_ONLY"}}
        cfg = dict(scope["make_config"](), encoder_channels=4, node_features=32,
                   gcn_hidden=48, image_size=32, neighbors=2, hop=3,
                   num_workers=0, amp=False, epochs=1)
        uninterrupted, resumed = root / "uninterrupted", root / "resumed"
        uninterrupted.mkdir()
        resumed.mkdir()
        fit = scope["fit_model"]
        fit(cfg, splits, torch.device("cpu"), resumed)
        cfg["epochs"] = 2
        fit(cfg, splits, torch.device("cpu"), resumed)
        fit(cfg, splits, torch.device("cpu"), uninterrupted)
        left = torch.load(resumed / "latest.pth", weights_only=True)
        right = torch.load(uninterrupted / "latest.pth", weights_only=True)
        assert left["epoch"] == right["epoch"] == 2
        for name, weight in left["model"].items():
            assert torch.equal(weight, right["model"][name]), name
        assert left["history"][1]["train_loss"] == right["history"][1]["train_loss"]
        invalid = dict(cfg, learning_rate=1e-3)
        try:
            fit(invalid, splits, torch.device("cpu"), resumed)
        except ValueError as error:
            assert "settings differ" in str(error)
        else:
            raise AssertionError("Incompatible checkpoint settings were accepted")
    print("Integration checks passed: full model, exact epoch-boundary resume, incompatible-setting rejection.")


if __name__ == "__main__":
    main()
