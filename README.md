# FRM × MTEM — Desktop Research Studio

[中文说明](README_中文.md)

FRM × MTEM is a local desktop application for liver fibrosis reversal prediction and treatment-response analysis. It brings together the clinical Fibrosis Reversal Model (FRM) and the Multimodal Treatment Effect Model (MTEM) in one interface. MTEM combines pretreatment clinical variables with paired H&E and Masson pathology features, using UNI2-h embeddings, cross-stain attention, and two treatment-specific prediction heads.

## Features

- Three analysis modes: FRM, MTEM, and combined FRM + MTEM.
- Graphical entry of baseline clinical measurements and Ishak stage, with automatic FIB-4 and APRI calculation.
- H&E / Masson pathology input through whole-slide images or pre-encoded UNI2-h features.
- FRM inference with the saved scikit-learn classifier; MTEM inference with the PyTorch two-head architecture.
- Checkpoint-specific preprocessing and probability calibration, with predictions displayed in the interface.
- Local inference and CSV result export.

## Repository structure

```text
FRM_MTEM_Desktop_App.py         Desktop application and GUI
deployment_runtime.py          Checkpoint loading and inference
model_architecture.py          MTEM network and feature validation
wsi_runtime.py                 WSI processing and UNI2-h encoding
research/training_pipeline.py  Model training and research analysis
scripts/check_deployment.py    Local model configuration check
requirements.txt               Base dependencies
requirements-wsi.txt           Optional WSI dependencies
models/                        Local model directory
tests/                         Automated tests
```

## Installation

Python 3.11 or 3.12 is recommended. From the project root, create a virtual environment and install the base dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For direct whole-slide image processing, install the optional dependencies:

```bash
python -m pip install -r requirements-wsi.txt
```

Start the application:

```bash
python scripts/check_deployment.py
python FRM_MTEM_Desktop_App.py
```

Windows users may also launch `start_windows.bat`.

## Model setup

Place compatible trained checkpoints in the local `models/` directory:

```text
models/
  FRM_selected_model.joblib
  MTEM444_fold1.pt
  MTEM444_fold1_preprocessing.joblib
  MTEM444_fold2.pt
  MTEM444_fold2_preprocessing.joblib
  MTEM444_fold3.pt
  MTEM444_fold3_preprocessing.joblib
  MTEM444_fold4.pt
  MTEM444_fold4_preprocessing.joblib
  MTEM444_fold5.pt
  MTEM444_fold5_preprocessing.joblib
```

All MTEM checkpoints and preprocessing files should come from the same compatible training run. Five folds are expected by default; `FRM_MTEM_EXPECTED_FOLDS` can be used to configure another fold count. Alternatively, set `FRM_MTEM_MODEL_DIR` to a local model directory.

This source repository does not bundle trained model weights. Compatible checkpoints can be generated using `research/training_pipeline.py` with appropriately formatted research datasets.

## Inputs

**Clinical:** Enter baseline measurements and Ishak stage in the GUI. FIB-4 and APRI are calculated automatically. Any additional trained clinical fields are handled using the saved preprocessing configuration and identified in the results.

**Pathology:** MTEM requires pretreatment H&E and Masson inputs from the same patient. Two input routes are available:

- **Encoded features:** Select matching UNI2-h `.npz` files. Patient identifiers, stain labels, and encoding metadata are checked automatically.
- **Whole-slide images:** Select supported WSI files such as `.svs`, `.tif`, or `.ndpi`. This route additionally requires OpenSlide, the UNI2-h encoder, and the training-time stain reference images at `models/reference-he.png` and `models/reference-masson.png`.

## Quick start

1. Launch the app and select FRM, MTEM, or combined analysis.
2. Enter baseline clinical inputs; for MTEM, select H&E and Masson files.
3. Run the analysis to view the FRM reversal probability and/or MTEM treatment-specific probabilities and their difference.
4. Export the displayed results to CSV.

## Development and tests

```bash
python -m compileall -q .
python -m pytest -q tests
```

Model artifacts, patient data, and pathology files are excluded from Git by `.gitignore`. Configure dependencies for your environment and follow applicable licenses for third-party tools and model assets.

This project is intended for research and software development.
