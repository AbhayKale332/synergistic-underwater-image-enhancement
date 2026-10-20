"""Compare FUGN with traditional and CNN baselines on the same UIEB split.

    modal run --env gpu --detach compare_methods.py
    modal volume get --force --env gpu underwater results/comparison results

Traditional methods run on the 256x256 evaluation images. The CNN baselines
use the FUGN loss, optimizer, crop pipeline, epochs and best-validation
checkpoint rule. FUGN is re-evaluated from the 10-epoch pilot checkpoint.
"""

import json
import os
from pathlib import Path
import shutil
import tempfile

import modal

MOUNT = "/mnt/underwater"
app = modal.App("fugn-uieb-comparison")
volume = modal.Volume.from_name("underwater")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libglib2.0-0", "unar")
    .pip_install_from_requirements("requirements.txt")
    .env({"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"})
    .add_local_file("FUGN_UIEB.ipynb", "/root/FUGN_UIEB.ipynb")
)

TRADITIONAL = ["Gray-World", "CLAHE", "UDCP", "MSRCR", "Fusion"]
DEEP = ["UWCNN-style", "U-Net"]
SHOWCASE = 6


def load_notebook():
    import nbformat
    notebook = nbformat.read("/root/FUGN_UIEB.ipynb", as_version=4)
    scope = {"__name__": "fugn_comparison"}
    for cell in notebook.cells:
        if {"definitions", "data-definitions"} & set(cell.metadata.get("tags", [])):
            exec(compile(cell.source, "FUGN_UIEB.ipynb", "exec"), scope)
    return scope


# ---------------------------------------------------------------- traditional
def gray_world(x):
    import numpy as np
    means = x.reshape(-1, 3).mean(0)
    return np.clip(x * (means.mean() / np.maximum(means, 1e-6)), 0, 1)


def clahe(x):
    import cv2
    import numpy as np
    lab = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32) / 255


def guided_filter(guide, source, radius=30, eps=1e-3):
    import cv2
    box = lambda a: cv2.boxFilter(a, -1, (2 * radius + 1,) * 2)
    mean_g, mean_s = box(guide), box(source)
    a = (box(guide * source) - mean_g * mean_s) / (box(guide * guide) - mean_g ** 2 + eps)
    b = mean_s - a * mean_g
    return box(a) * guide + box(b)


def udcp(x, patch=15, omega=0.95, t_min=0.1):
    # Drews et al. underwater dark channel: only the green and blue channels.
    import cv2
    import numpy as np
    kernel = np.ones((patch, patch), np.uint8)
    dark = cv2.erode(x[..., 1:].min(-1), kernel)
    brightest = np.argsort(dark.ravel())[-max(1, dark.size // 1000):]
    airlight = x.reshape(-1, 3)[brightest].mean(0)
    normalized = x[..., 1:] / np.maximum(airlight[1:], 1e-6)
    transmission = 1 - omega * cv2.erode(normalized.min(-1), kernel)
    gray = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2GRAY)
    transmission = guided_filter(gray, transmission.astype(np.float32))
    restored = (x - airlight) / np.maximum(transmission, t_min)[..., None] + airlight
    return np.clip(restored, 0, 1)


def msrcr(x, sigmas=(15, 80, 250), alpha=125.0, beta=46.0):
    import cv2
    import numpy as np
    image = x.astype(np.float64) * 255 + 1
    retinex = sum(np.log(image) - np.log(cv2.GaussianBlur(image, (0, 0), s)) for s in sigmas) / len(sigmas)
    restoration = beta * (np.log(alpha * image) - np.log(image.sum(-1, keepdims=True)))
    result = restoration * retinex
    out = np.empty_like(result)
    for c in range(3):  # 1% simplest colour balance per channel
        low, high = np.percentile(result[..., c], (1, 99))
        out[..., c] = (result[..., c] - low) / max(high - low, 1e-6)
    return np.clip(out, 0, 1).astype(np.float32)


def fusion(x, levels=5):
    # Ancuti et al. (2012): white-balanced and contrast-enhanced inputs blended
    # with Laplacian-contrast, saliency and saturation weights in pyramids.
    import cv2
    import numpy as np
    first = gray_world(x).astype(np.float32)
    second = clahe(first)
    def weights(img):
        lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = np.abs(cv2.Laplacian(lum, cv2.CV_32F))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        blurred = cv2.GaussianBlur(lab, (5, 5), 0)
        saliency = np.linalg.norm(blurred - lab.reshape(-1, 3).mean(0), axis=-1)
        saturation = np.sqrt(((img - lum[..., None]) ** 2).mean(-1))
        return laplacian / (laplacian.max() + 1e-6) + saliency / (saliency.max() + 1e-6) + saturation
    w = np.stack([weights(first), weights(second)])
    w = w / (w.sum(0) + 1e-6)
    def gaussian_pyramid(img):
        pyramid = [img]
        for _ in range(levels - 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        return pyramid
    def laplacian_pyramid(img):
        g = gaussian_pyramid(img)
        return [g[i] - cv2.pyrUp(g[i + 1], dstsize=g[i].shape[1::-1]) for i in range(levels - 1)] + [g[-1]]
    blended = None
    for img, weight in zip((first, second), w):
        gw, li = gaussian_pyramid(weight.astype(np.float32)), laplacian_pyramid(img)
        layers = [l * g[..., None] for l, g in zip(li, gw)]
        blended = layers if blended is None else [a + b for a, b in zip(blended, layers)]
    out = blended[-1]
    for layer in reversed(blended[:-1]):
        out = cv2.pyrUp(out, dstsize=layer.shape[1::-1]) + layer
    return np.clip(out, 0, 1)


# ---------------------------------------------------------------- deep baselines
def build_baselines(nn, F):
    class UWCNN(nn.Module):
        # Plain residual CNN in the spirit of UWCNN: no graph, no downsampling.
        def __init__(self, channels=64, depth=10):
            super().__init__()
            layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True)]
            for _ in range(depth - 2):
                layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
            layers.append(nn.Conv2d(channels, 3, 3, padding=1))
            self.body = nn.Sequential(*layers)

        def forward(self, x, adjacency=None):
            return x + self.body(x)

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1))

    class UNet(nn.Module):
        def __init__(self, base=32):
            super().__init__()
            widths = [base, base * 2, base * 4, base * 8]
            self.downs = nn.ModuleList()
            previous = 3
            for w in widths:
                self.downs.append(block(previous, w))
                previous = w
            self.bottleneck = block(widths[-1], widths[-1] * 2)
            self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
            previous = widths[-1] * 2
            for w in reversed(widths):
                self.ups.append(nn.ConvTranspose2d(previous, w, 2, stride=2))
                self.decoders.append(block(w * 2, w))
                previous = w
            self.final = nn.Conv2d(base, 3, 1)

        def forward(self, x, adjacency=None):
            skips, h = [], x
            for down in self.downs:
                h = down(h)
                skips.append(h)
                h = F.max_pool2d(h, 2)
            h = self.bottleneck(h)
            for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
                h = dec(torch_cat(up(h), skip))
            return x + self.final(h)

    def torch_cat(a, b):
        import torch
        return torch.cat([a, b], 1)

    return {"UWCNN-style": UWCNN, "U-Net": UNet}


def train_baseline(nb, name, model, cfg, splits, device, output):
    import math
    import time
    import numpy as np
    import torch
    import piq
    nb["seed_everything"](cfg["seed"])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", init_scale=cfg["amp_initial_scale"])
    train_data = nb["PairedImages"](splits["train"], cfg, training=True)
    val_loader = nb["make_loader"](nb["PairedImages"](splits["val"], cfg), cfg["batch_size"])
    history, best_psnr, best_state = [], -math.inf, None
    for epoch in range(1, cfg["epochs"] + 1):
        started = time.perf_counter()
        train_data.epoch = epoch
        loader = nb["make_loader"](train_data, cfg["batch_size"], shuffle=True, seed=cfg["seed"] + epoch)
        model.train()
        total, seen = 0.0, 0
        for raw, reference, _ in loader:
            raw, reference = raw.to(device), reference.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                prediction = model(raw)
            loss = nb["loss_components"](prediction.float(), reference, cfg)[0]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{name}: non-finite loss at epoch {epoch}")
            nb["optimizer_update"](loss, model, optimizer, scaler, cfg["grad_clip"])
            total += loss.item() * raw.shape[0]
            seen += raw.shape[0]
        model.eval()
        psnr, ssim = [], []
        with torch.inference_mode():
            for raw, reference, _ in val_loader:
                raw, reference = raw.to(device), reference.to(device)
                clipped = model(raw).float().clamp(0, 1)
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
    torch.save(best_state, output / f"{name}.pth")
    return model.eval(), history


# ---------------------------------------------------------------- metrics
def classification_metrics(prediction, reference, tolerance=0.1):
    """Pixel accuracy and 1-pixel-tolerant Canny edge precision/recall/F1."""
    import cv2
    import numpy as np
    accuracy = float((np.abs(prediction - reference).mean(-1) <= tolerance).mean())
    def edges(rgb):
        gray = cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150) > 0
    predicted, target = edges(prediction), edges(reference)
    kernel = np.ones((3, 3), np.uint8)
    near_target = cv2.dilate(target.astype(np.uint8), kernel) > 0
    near_predicted = cv2.dilate(predicted.astype(np.uint8), kernel) > 0
    precision = float((predicted & near_target).sum() / max(predicted.sum(), 1))
    recall = float((target & near_predicted).sum() / max(target.sum(), 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1}


@app.function(image=image, gpu="A10G", volumes={MOUNT: volume}, timeout=4 * 3600, cpu=8, memory=32768)
def compare(epochs: int = 10):
    import time
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch import nn
    from PIL import Image

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
    pilot_cfg = json.loads((Path(MOUNT) / "results" / "pilot" / "config.json").read_text())
    assert all(cfg[k] == pilot_cfg[k] for k in ("seed", "learning_rate", "batch_size", "image_size", "epochs"))
    splits = nb["prepare_data"](cfg)
    output = Path(MOUNT) / "results" / "comparison"
    (output / "images").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    models, histories = {}, []
    for name, factory in build_baselines(nn, F).items():
        model, history = train_baseline(nb, name, factory(), cfg, splits, device, output)
        models[name] = model
        histories += history
    fugn = nb["FUGN"](cfg).to(device)
    checkpoint = torch.load(Path(MOUNT) / "results" / "pilot" / "best.pth", map_location=device, weights_only=True)
    fugn.load_state_dict(checkpoint["model"])
    models["FUGN"] = fugn.eval()
    histories += [{"method": "FUGN", "epoch": r["epoch"], "train_loss": r["train_loss"],
                   "val_psnr": r["val_psnr"], "val_ssim": r["val_ssim"], "seconds": r["seconds"]}
                  for r in checkpoint["history"]]
    pd.DataFrame(histories).to_csv(output / "training_history.csv", index=False)
    parameters = {name: sum(p.numel() for p in m.parameters()) for name, m in models.items()}

    traditional = {"Gray-World": gray_world, "CLAHE": clahe, "UDCP": udcp, "MSRCR": msrcr, "Fusion": fusion}
    rows = []
    for group in ("test", "challenging"):
        dataset = nb["PairedImages"](splits[group], cfg)
        for index in range(len(dataset)):
            raw, reference, name = dataset[index]
            raw_np = raw.permute(1, 2, 0).numpy()
            outputs, timings = {"Input": raw_np}, {"Input": 0.0}
            for method, function in traditional.items():
                start = time.perf_counter()
                outputs[method] = function(raw_np).astype(np.float32)
                timings[method] = (time.perf_counter() - start) * 1000
            batch = raw[None].to(device)
            for method, model in models.items():
                with torch.inference_mode():
                    graph = nb["build_graph"](batch, cfg) if method == "FUGN" else None
                    model(batch, graph)  # warm-up
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    graph = nb["build_graph"](batch, cfg) if method == "FUGN" else None
                    prediction = model(batch, graph).float().clamp(0, 1)
                    torch.cuda.synchronize()
                timings[method] = (time.perf_counter() - start) * 1000
                outputs[method] = prediction[0].cpu().permute(1, 2, 0).numpy()
            ref_np = reference.permute(1, 2, 0).numpy() if reference.numel() else None
            for method, rgb in outputs.items():
                row = {"split": group, "image": name, "method": method, "ms": timings[method]}
                if ref_np is not None:
                    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].to(device)
                    full = nb["full_reference_metrics"](tensor.clamp(0, 1), reference[None].to(device))
                    row.update({k: v.item() for k, v in full.items()})
                    row["MSE"] = float(((rgb - ref_np) ** 2).mean())
                    row.update(classification_metrics(rgb, ref_np))
                row.update(nb["no_reference_metrics"](rgb))
                rows.append(row)
            if index < SHOWCASE or (group == "challenging" and index < 4):
                folder = output / "images" / group / name
                folder.mkdir(parents=True, exist_ok=True)
                for method, rgb in outputs.items():
                    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(folder / f"{method}.png")
                if ref_np is not None:
                    Image.fromarray(np.rint(ref_np * 255).astype(np.uint8)).save(folder / "Reference.png")
            if (index + 1) % 15 == 0:
                print(f"{group}: {index + 1}/{len(dataset)}", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)

    # Intermediate FUGN stages for the first two test photographs.
    stages = {}
    for raw, _, name in [nb["PairedImages"](splits["test"], cfg)[i] for i in range(2)]:
        batch = raw[None].to(device)
        with torch.inference_mode():
            adjacency, similarity = nb["build_graph"](batch, cfg, return_similarity=True)
            encoded = fugn.encoder(batch)
            down = fugn.down(encoded)
            nodes = nb["features_to_nodes"](down)
            norm = nb["normalize_graph"](adjacency)
            hidden = F.relu(fugn.gcn1(nodes, norm))
            gcn = nb["nodes_to_features"](fugn.gcn2(hidden, norm), 16, 16)
            decoded = fugn.decoder(gcn + down)
            final = fugn.final(decoded + encoded).clamp(0, 1)
            gray = (batch * batch.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
            sobel = batch.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None]
            edge = torch.sqrt(F.conv2d(gray, sobel, padding=1) ** 2 + F.conv2d(gray, sobel.transpose(-1, -2), padding=1) ** 2)
        stages[name] = {
            "edges": edge[0, 0].cpu().numpy().tolist(),
            "similarity": similarity[0].cpu().numpy().tolist(),
            "adjacency": adjacency[0].cpu().numpy().tolist(),
            "encoder": encoded[0].abs().mean(0).cpu().numpy().tolist(),
            "down": down[0].norm(dim=0).cpu().numpy().tolist(),
            "gcn": gcn[0].norm(dim=0).cpu().numpy().tolist(),
            "decoder": decoded[0].abs().mean(0).cpu().numpy().tolist(),
            "output": final[0].permute(1, 2, 0).cpu().numpy().tolist(),
        }
    (output / "stages.json").write_text(json.dumps(stages))
    (output / "summary.json").write_text(json.dumps({
        "epochs": epochs, "parameters": parameters, "learning_rate": cfg["learning_rate"],
        "batch_size": cfg["batch_size"], "seed": cfg["seed"],
        "gpu": torch.cuda.get_device_name(0)}, indent=2))
    volume.commit()
    summary = metrics[metrics.split == "test"].groupby("method").mean(numeric_only=True)
    print(summary.round(4).to_string(), flush=True)


@app.local_entrypoint()
def main(epochs: int = 10):
    compare.remote(epochs)
