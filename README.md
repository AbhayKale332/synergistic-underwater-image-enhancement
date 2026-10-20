# Synergistic underwater image enhancement · FUGN

An executable implementation of **Fusion-based graph neural networks for
synergistic underwater image enhancement** (Information Fusion 117, 2025,
102857), with a real-image UIEB pilot and an offline HTML results report.

Open [`FUGN_UIEB.ipynb`](FUGN_UIEB.ipynb) for the methodology, equations,
implementation, training, evaluation, graphs, and all comparisons.

### Hosted results

[**View the results report**](https://recova.indiasouthcentral.cloudapp.azure.com/underwater-results/)

The completed **10-epoch pilot** report includes all **150 input/output
comparisons** and **six diagnostic plots**. It is hosted on the existing
Azure `recova-vm` through Caddy with HTTPS and gzip compression. The initial
download is about 91 MB because all photographs and plots are embedded.

The HTML and its compressed copy persist in Caddy's data volume at
`/data/research/fugn/index.html` and `index.html.gz`. The same standalone HTML
can be downloaded from Modal and opened offline using the instructions below.

[**View the method comparison**](https://recova.indiasouthcentral.cloudapp.azure.com/underwater-results/comparison/)
(about 11 MB) is stored beside it at `/data/research/fugn/comparison/`.

### Method comparison

[`compare_methods.py`](compare_methods.py) evaluates FUGN against five
traditional methods and two CNN baselines on the **same 90 paired test and 60
challenging photographs**, with the notebook's own split, resizing and metric
code. Traditional methods: Gray-World, CLAHE (LAB), UDCP (Drews et al.),
MSRCR Retinex and Fusion (Ancuti et al. 2012). The CNN baselines, a
UWCNN-style residual CNN (298,947 parameters) and a U-Net (7,766,051), use
FUGN's loss, optimizer, learning rate, batch size, crops, 10 epochs and
best-validation checkpoint rule. FUGN is the pilot checkpoint, re-evaluated;
its scores match the pilot exactly.

| Method | PSNR (dB) | SSIM | FSIM | Pixel accuracy | Edge F1 | PSNR wins vs input |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untreated input | 17.77 | 0.7737 | 0.8924 | 0.5229 | 0.7094 | – |
| Gray-World | 17.52 | 0.7721 | 0.8739 | 0.5187 | 0.7273 | 50/90 |
| CLAHE | 18.02 | 0.8257 | 0.9400 | 0.5269 | **0.8803** | 56/90 |
| UDCP | 14.20 | 0.6921 | 0.9220 | 0.2789 | 0.8283 | 14/90 |
| MSRCR | 15.25 | 0.7790 | 0.8913 | 0.3340 | 0.8262 | 37/90 |
| Fusion | 18.35 | 0.8362 | 0.9191 | 0.5573 | 0.8477 | 56/90 |
| UWCNN-style CNN | 19.73 | 0.8685 | 0.9445 | 0.6504 | 0.8705 | 72/90 |
| U-Net | 18.88 | 0.8505 | 0.9412 | 0.6048 | 0.8575 | 63/90 |
| **FUGN** | **19.93** | **0.8721** | **0.9476** | **0.6687** | 0.8771 | **74/90** |

Pixel accuracy is the share of pixels whose mean absolute RGB error from the
reference is at most 0.1. Edge precision, recall and F1 compare Canny edges
(50/150) of output and reference with 1-pixel tolerance. FUGN has the best
PSNR, SSIM, FSIM, MSE (0.0149) and pixel accuracy; CLAHE has the best edge
recall and F1 but the lowest edge precision among enhanced outputs.
UDCP and Gray-World have the highest UCIQE and UIQM even though their PSNR is
below the input, which shows why no-reference scores need visual inspection.
The U-Net produced red blotch artifacts on several photographs, most likely
from BatchNorm at batch size 2. FUGN's margin over the UWCNN-style CNN
(0.20 dB) is from a single seed and a 10-epoch budget, so it is not an
established difference.

```bash
modal run --env gpu --detach compare_methods.py
modal volume get --force --env gpu underwater results/comparison results
```

The run (about 12 minutes on an A10G) writes `metrics.csv` (per image, per
method), `training_history.csv`, `summary.json`, baseline checkpoints,
showcase PNGs and FUGN stage activations (`stages.json`) to
`/mnt/underwater/results/comparison`. It reads the pilot's `best.pth`, so run
the pilot first.

### Run the notebook

Use Python 3.12 and a CUDA GPU (the full model has 31,183,007 parameters).
On Debian/Ubuntu, install `unar` (`sudo apt-get install unar`) to read the
official password-protected RAR downloads; ZIP archives are supported too.
The notebook uses the public extraction passwords published by the UIEB authors.

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name python3
```

Open the notebook in Jupyter or VS Code and run all cells. The default is a
**10-epoch pilot**, Adam learning rate **2e-4**, and batch size 2. The `paper`
preset selects the PDF learning rate **1e-6** and 100 epochs:

```bash
export FUGN_PRESET=paper
export FUGN_OUTPUT_DIR=results/paper
```

Other environment settings: `FUGN_DATA_DIR` (default `data/UIEB`),
`FUGN_OUTPUT_DIR` (default `results/pilot`), `FUGN_EPOCHS` (total epoch limit),
and `FUGN_RESUME` (`1` by default). Use a new output directory when changing
training settings. Increasing the epoch limit resumes the same experiment.

### Run on Modal through GitHub

The launcher allocates CPU for downloads and then an **A10G GPU** for execution.
It clones the public repository and checks out the specified immutable commit.
The existing `underwater` volume is mounted at **`/mnt/underwater`**, with
`data/UIEB`, `source/<commit>`, and `results/<run-name>` beneath it. Source comes
from GitHub; datasets and output artifacts stay on the volume.
For execution, archives are copied sequentially to temporary GPU-local scratch
and extracted there, avoiding slow cold reads of thousands of individual files.
The canonical data and all outputs remain on `/mnt/underwater`. A validated
split manifest is reused when filenames and byte sizes are unchanged; set
`FUGN_VERIFY_DATA=1` to force a fresh image-validation pass.

Install the Modal CLI separately (`pip install modal`) and authenticate, then:

```bash
modal run --env gpu --detach modal_run.py --commit FULL_40_CHARACTER_GIT_SHA
```

Download the report and results after completion:

```bash
mkdir -p results
modal volume get --force --env gpu underwater results/pilot results
python check_notebook.py --artifacts results/pilot
```

To prepare only the data, add `--prepare-only`. To extend the same pilot, pass
`--epochs 100`. To start a separate PDF-settings run, pass
`--preset paper --epochs 100 --run-name paper`.

### What the experiment produces

- A standalone **`fugn_uieb_pilot.html`** with embedded photographs and plots;
  it opens offline without a server or JavaScript dependencies.
- **90 held-out input / output / reference triplets** and **60 challenging
  input / output pairs**, all included without selection by quality.
- Loss, validation PSNR/SSIM, input/output metric charts, per-image improvement
  distributions, representative RGB histograms, and graph diagnostics.
- An executed notebook; best/latest resumable checkpoints; 150 enhanced PNGs;
  30 comparison sheets; per-image and aggregate CSVs; configuration, software
  versions, source commit, and a fixed split manifest.

UIEB is downloaded from its authors' official links (about 1.4 GB for the paired
archives, plus the challenging set). The split is **720 train / 80 validation /
90 paired test**, seed 42, plus 60 unpaired challenging tests. Training uses
aligned random crops; evaluation resizes to 256×256. PSNR, SSIM and genuine
phase-congruency FSIM are computed only for paired tests. Both sets receive
UIQM and UCIQE scores, also measured on untreated input photographs.

The completed 10-epoch pilot (source commit `24fcac7`, seed 42) measured the
following on the 90 held-out paired photographs:

| Metric | Untreated input | FUGN |
| --- | ---: | ---: |
| PSNR (dB) | 17.77 | 19.93 |
| SSIM | 0.7737 | 0.8721 |

PSNR improved on 74/90 images. All 3,600 optimizer updates succeeded without
overflow skips. The report is available at the hosted link above and on the
`underwater` volume at `/mnt/underwater/results/pilot/fugn_uieb_pilot.html`.

### Fidelity and interpretation

This is an implementation of the methodology, **not a claim of reproducing the
paper's benchmark**. The initial run is one ten-epoch pilot, while the paper
reports ten-run averages. The PDF's exact paired split and numeric graph
bandwidths are unavailable. It describes 60 test photographs that the UIEB
authors identify as reference-free, so this implementation reserves separate
paired tests for reference-based evaluation.

Explicit graph defaults are k=5, h=255 (global candidate search), Gaussian
7×7/σ=1, and separate per-image median-distance RBF scales. The author code's
encoder residual block and dropout 0.4 are retained. The GCN reshape is
corrected; graphs follow the exact cropped input batch; RGB floating-point
operator descriptors replace clipped unsigned-byte/BGR processing. Final
64-channel encoder skips are fused before the RGB output convolution.

The loss follows the PDF's equal Charbonnier/SSIM weighting. The pilot uses
the code's faster learning rate. Metric definitions and scaling conventions
are stated in the notebook and report, especially for UIQM/UCIQE. No-reference
scores can reward artifacts; inspect the comparisons as well as averages.
CUDA uses mixed precision with automatic gradient scaling (initial scale 1024).
Any overflow-skipped updates are recorded alongside successful optimizer updates
in the history and report summary; non-finite losses still stop training.

### Checks

```bash
python check_notebook.py
```

This runs the actual notebook definitions and checks the full model's shape
and parameter count, graph/crop alignment, batch isolation, spatial reshape,
finite gradients, metric identities, and exact resumed-vs-uninterrupted
training on temporary synthetic **test fixtures only**. Those fixtures never
enter the real-image experiment or report.

### Sources and dataset use

- [FUGN paper](https://doi.org/10.1016/j.inffus.2024.102857)
- [FUGN author code](https://github.com/whZ62/FUGN)
- [UIEB dataset authors and download terms](https://li-chongyi.github.io/proj_benchmark.html)
- [PIQ metric implementation](https://piq.readthedocs.io/en/latest/functions.html)

UIEB is restricted to academic use and its authors prohibit redistribution.
Downloaded images, executed notebooks containing dataset images, reports,
checkpoints and logs are ignored by Git. Do not commit these artifacts to the
public source repository. The PDF itself is not redistributed here.
