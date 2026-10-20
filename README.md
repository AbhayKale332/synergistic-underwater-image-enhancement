# Synergistic underwater image enhancement · FUGN

An executable implementation of **Fusion-based graph neural networks for
synergistic underwater image enhancement** (Information Fusion 117, 2025,
102857), with a real-image UIEB pilot and an offline HTML results report.

Open [`FUGN_UIEB.ipynb`](FUGN_UIEB.ipynb) for the methodology, equations,
implementation, training, evaluation, graphs, and all comparisons.

### Run the notebook

Use Python 3.12 and a CUDA GPU (the full model has 31,183,007 parameters).

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

Install the Modal CLI separately (`pip install modal`) and authenticate, then:

```bash
modal run --env gpu --detach modal_run.py --commit FULL_40_CHARACTER_GIT_SHA
```

Download the report and results after completion:

```bash
modal volume get --env gpu underwater results/pilot ./results/pilot
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
