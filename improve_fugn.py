"""FUGN-R: FUGN's operator graph on a pretrained residual U-Net, 10 epochs on UIEB.

    modal run --env gpu --detach improve_fugn.py
    modal volume get --force --env gpu underwater results/improved results

Same 720/80/90 split, 60 challenging images, metric code, 10-epoch budget
and best-validation-PSNR checkpoint rule as the FUGN pilot and the baseline
comparison. Variants are ranked on validation PSNR only; test images are
scored once, after selection.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile

import modal

MOUNT = "/mnt/underwater"
app = modal.App("fugn-uieb-improved")
volume = modal.Volume.from_name("underwater")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libglib2.0-0", "unar")
    .pip_install_from_requirements("requirements.txt")
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    .add_local_file("FUGN_UIEB.ipynb", "/root/FUGN_UIEB.ipynb")
    .add_local_python_source("compare_methods")
)

# name: (ResNet encoder, graph: "operator" = FUGN Sobel/Gaussian blocks,
# "feature" = k-NN on encoder features, None = no graph, training views)
VARIANTS = {
    "FUGN-R34": ("resnet34", "operator", "mixed"),
    "FUGN-R34 (crops only)": ("resnet34", "operator", "crops"),
    "FUGN-R34 (feature graph)": ("resnet34", "feature", "mixed"),
    "R34 U-Net (no graph)": ("resnet34", None, "mixed"),
    "FUGN-R50": ("resnet50", "operator", "mixed"),
    "R50 U-Net (no graph)": ("resnet50", None, "mixed"),
}


def build_model(nn, F, torch, torchvision, nb, cfg, encoder, graph):
    def conv_block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())

    class GraphBlock(nn.Module):
        # FUGN's two-layer GCN over the 16x16 block graph, fused residually.
        def __init__(self, channels, hidden):
            super().__init__()
            self.norm = nn.LayerNorm(channels)
            self.gcn1 = nb["GraphConv"](channels, hidden)
            self.gcn2 = nb["GraphConv"](hidden, channels)
            self.scale = nn.Parameter(torch.zeros(1))

        def forward(self, features, adjacency):
            nodes = self.norm(nb["features_to_nodes"](features))
            if adjacency is None:  # feature graph: k most cosine-similar blocks
                with torch.no_grad():
                    unit = F.normalize(nodes.float(), dim=-1)
                    similarity = unit @ unit.transpose(1, 2)
                    similarity.diagonal(dim1=1, dim2=2).fill_(-torch.inf)
                    neighbors = similarity.topk(cfg["neighbors"], dim=-1).indices
                    adjacency = torch.zeros_like(similarity).scatter_(-1, neighbors, 1.0)
            norm = nb["normalize_graph"](adjacency).to(nodes.dtype)
            hidden = F.dropout(F.gelu(self.gcn1(nodes, norm)), 0.1, self.training)
            update = nb["nodes_to_features"](self.gcn2(hidden, norm), *features.shape[-2:])
            return features + self.scale * update

    class FUGNR(nn.Module):
        def __init__(self):
            super().__init__()
            resnet = getattr(torchvision.models, encoder)(weights="IMAGENET1K_V1")
            c = [64, 64, 128, 256, 512] if encoder == "resnet34" else [64, 256, 512, 1024, 2048]
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 1/2
            self.pool = resnet.maxpool
            self.layers = nn.ModuleList([resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4])
            self.full = conv_block(3, 32)  # full-resolution skip
            self.graph = GraphBlock(c[3], 512) if graph else None  # layer3: 16x16 blocks
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
            # (incoming, skip, out) from 1/32 up to full resolution
            spec = [(c[4], c[3], 256), (256, c[2], 128), (128, c[1], 64), (64, c[0], 64), (64, 32, 32)]
            self.decoders = nn.ModuleList(conv_block(i + s, o) for i, s, o in spec)
            self.head = nn.Conv2d(32, 3, 3, padding=1)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def train(self, mode=True):
            # Batch size 4 is too small for BatchNorm statistics: keep ImageNet ones.
            super().train(mode)
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            return self

        def forward(self, x, adjacency=None):
            h = self.stem((x - self.mean) / self.std)
            skips = [self.full(x), h]
            h = self.pool(h)
            for index, layer in enumerate(self.layers):
                h = layer(h)
                if index == 2 and self.graph is not None:
                    h = self.graph(h, adjacency)
                if index < 3:
                    skips.append(h)
            for decoder, skip in zip(self.decoders, reversed(skips)):
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                h = decoder(torch.cat([h, skip], 1))
            return x + self.head(h)

    return FUGNR()


def load_pairs(read_rgb, records, size):
    import numpy as np
    from PIL import Image
    resize = lambda path: np.asarray(read_rgb(path).resize((size, size), Image.Resampling.BICUBIC))
    return (np.stack([resize(raw) for raw, _, _ in records]),
            np.stack([resize(ref) for _, ref, _ in records]))


def train_variant(nb, name, encoder, graph_mode, views, cfg, splits, device, output):
    import math
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchvision
    from torch import nn
    import piq

    nb["seed_everything"](cfg["seed"])
    model = build_model(nn, F, torch, torchvision, nb, cfg, encoder, graph_mode).to(device)
    use_graph = graph_mode == "operator"
    pretrained = [p for n, p in model.named_parameters() if n.startswith(("stem", "layers"))]
    fresh = [p for n, p in model.named_parameters() if not n.startswith(("stem", "layers"))]
    epochs, batch = cfg["epochs"], 4
    steps = epochs * math.ceil(len(splits["train"]) / batch)
    optimizer = torch.optim.AdamW([{"params": pretrained, "lr": 2e-4},
                                   {"params": fresh, "lr": 1e-3}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[2e-4, 1e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", init_scale=cfg["amp_initial_scale"])

    # Full images resized exactly like evaluation, plus random-resized crops from 384px copies.
    full_raw, full_ref = (torch.from_numpy(a).permute(0, 3, 1, 2) for a in load_pairs(nb["read_rgb"], splits["train"], 256))
    big_raw, big_ref = (torch.from_numpy(a).permute(0, 3, 1, 2) for a in load_pairs(nb["read_rgb"], splits["train"], 384))
    val = nb["PairedImages"](splits["val"], cfg)
    val_raw = torch.stack([val[i][0] for i in range(len(val))])
    val_ref = torch.stack([val[i][1] for i in range(len(val))])
    generator = torch.Generator().manual_seed(cfg["seed"])

    def sample(indices):
        raws, refs = [], []
        for i in indices.tolist():
            crop = views == "crops" or (views == "mixed" and torch.rand(1, generator=generator).item() < 0.5)
            if crop:
                side = int(torch.randint(192, 385, (1,), generator=generator))
                top, left = (int(torch.randint(0, 385 - side, (1,), generator=generator)) for _ in range(2))
                pair = torch.stack([big_raw[i], big_ref[i]])[..., top:top + side, left:left + side].float()
                pair = F.interpolate(pair, size=(256, 256), mode="bicubic", align_corners=False, antialias=True)
            else:
                pair = torch.stack([full_raw[i], full_ref[i]]).float()
            if torch.rand(1, generator=generator).item() < 0.5:
                pair = pair.flip(-1)
            raws.append(pair[0])
            refs.append(pair[1])
        return (torch.stack(raws) / 255).clamp(0, 1), (torch.stack(refs) / 255).clamp(0, 1)

    history, best_psnr, best_state, updates = [], -math.inf, None, 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        total, seen = 0.0, 0
        for indices in torch.randperm(len(full_raw), generator=generator).split(batch):
            raw, reference = (t.to(device) for t in sample(indices))
            graph = nb["build_graph"](raw, cfg) if use_graph else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                prediction = model(raw, graph)
            loss = nb["loss_components"](prediction.float(), reference, cfg)[0]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: non-finite loss at epoch {epoch}")
            nb["optimizer_update"](loss, model, optimizer, scaler, 1.0)
            scheduler.step()
            updates += 1
            total += loss.item() * raw.shape[0]
            seen += raw.shape[0]
        model.eval()
        psnr, ssim = [], []
        with torch.inference_mode():
            for raw, reference in zip(val_raw.split(8), val_ref.split(8)):
                raw, reference = raw.to(device), reference.to(device)
                graph = nb["build_graph"](raw, cfg) if use_graph else None
                clipped = model(raw, graph).float().clamp(0, 1)
                psnr += (-10 * torch.log10((clipped - reference).square().flatten(1).mean(1))).tolist()
                ssim += piq.ssim(clipped, reference, downsample=False, reduction="none").tolist()
        row = {"method": name, "epoch": epoch, "train_loss": total / seen,
               "val_psnr": float(np.mean(psnr)), "val_ssim": float(np.mean(ssim)),
               "seconds": time.perf_counter() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_psnr"] > best_psnr:
            best_psnr = row["val_psnr"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, output / f"{name.replace(' ', '_')}.pth")
    params = sum(p.numel() for p in model.parameters())
    return model.eval(), history, best_psnr, params, updates


@app.function(image=image, gpu="A10G", volumes={MOUNT: volume}, timeout=4 * 3600, cpu=8, memory=32768)
def improve(epochs: int = 10):
    import time
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from compare_methods import classification_metrics, load_notebook

    nb = load_notebook()
    volume.reload()
    persistent = Path(MOUNT) / "data" / "UIEB"
    working = Path(tempfile.mkdtemp(prefix="uieb-"))
    for subset in ("raw", "reference", "challenging"):
        shutil.copyfile(persistent / f"{subset}.archive", working / f"{subset}.archive")
    for manifest in persistent.glob("split_seed_*.json"):
        shutil.copyfile(manifest, working / manifest.name)
    os.environ.update({"FUGN_DATA_DIR": str(working), "FUGN_EPOCHS": str(epochs)})
    cfg = nb["make_config"]()
    splits = nb["prepare_data"](cfg)
    output = Path(MOUNT) / "results" / "improved"
    (output / "images").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    trained, histories = {}, []
    for name, (encoder, graph_mode, views) in VARIANTS.items():
        model, history, best, params, updates = train_variant(
            nb, name, encoder, graph_mode, views, cfg, splits, device, output)
        trained[name] = {"model": model, "graph": graph_mode == "operator", "val_psnr": best,
                         "encoder": encoder, "graph_mode": graph_mode, "views": views,
                         "parameters": params, "updates": updates}
        histories += history
    pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False)
    selected = max(trained, key=lambda n: trained[n]["val_psnr"])
    print("Selected on validation PSNR:", selected, flush=True)

    rows = []
    for group in ("test", "challenging"):
        dataset = nb["PairedImages"](splits[group], cfg)
        for index in range(len(dataset)):
            raw, reference, name = dataset[index]
            batch = raw[None].to(device)
            ref_np = reference.permute(1, 2, 0).numpy() if reference.numel() else None
            for method, entry in trained.items():
                model = entry["model"]
                with torch.inference_mode():
                    model(batch, nb["build_graph"](batch, cfg) if entry["graph"] else None)  # warm-up
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    graph = nb["build_graph"](batch, cfg) if entry["graph"] else None
                    prediction = model(batch, graph).float().clamp(0, 1)
                    torch.cuda.synchronize()
                ms = (time.perf_counter() - start) * 1000
                rgb = prediction[0].cpu().permute(1, 2, 0).numpy()
                row = {"split": group, "image": name, "method": method, "ms": ms}
                if ref_np is not None:
                    full = nb["full_reference_metrics"](prediction, reference[None].to(device))
                    row.update({k: v.item() for k, v in full.items()})
                    row["MSE"] = float(((rgb - ref_np) ** 2).mean())
                    row.update(classification_metrics(rgb, ref_np))
                row.update(nb["no_reference_metrics"](rgb))
                rows.append(row)
                if method == selected:
                    folder = output / "images" / group
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(folder / f"{name}.png")
            if (index + 1) % 15 == 0:
                print(f"{group}: {index + 1}/{len(dataset)}", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    (output / "summary.json").write_text(json.dumps({
        "epochs": epochs, "selected": selected, "seed": cfg["seed"], "batch_size": 4,
        "gpu": torch.cuda.get_device_name(0),
        "variants": {n: {k: e[k] for k in ("encoder", "graph_mode", "views", "val_psnr", "parameters", "updates")}
                     for n, e in trained.items()}}, indent=2))
    volume.commit()
    for group in ("test", "challenging"):
        print(group, metrics[metrics.split == group].groupby("method").mean(numeric_only=True)
              .round(4).to_string(), sep="\n", flush=True)


@app.local_entrypoint()
def main(epochs: int = 10):
    improve.remote(epochs)
