# Simple implementations of different types of Autoencoders in PyTorch

## Overview

This repository contains PyTorch implementations of different type of autoencoders.

**Try the interactive demo:** https://julienexr-autoencooder-mnist-app-sehuso.streamlit.app/

## Table of contents

- [Quick project layout](#quick-project-layout)
- [Modules](#modules)
- [Usage instructions](#usage-instructions)
- [Small tips](#small-tips)

## Quick project layout

- `modules/autoencoder/ae.py` -- AE model (encoder + decoder).
- `modules/autoencoder/training.py` -- AE training loop.
- `modules/vae/vae.py` -- VAE model (probabilistic encoder producing μ and logvar + decoder). Two encoder/decoder variants exist (`default` and `pp`).
- `modules/vae/training.py` -- VAE training loop.
- `modules/cvae/cvae.py` -- CVAE model (VAE conditioned on class labels).
- `modules/cvae/training.py` -- CVAE training loop.
- `main.py` -- example entry points. By default it runs AE training then VAE training (see note below).
- `src/visualization.py` -- `Visualizer` helper used by `training.py` to save reconstructions, PCA plots, interpolations and noise samples into `visu/`.
- `src/data.py` -- MNIST dataloader helpers.
- `modules/autoencoder/` -- AE module README and usage notes.
- `modules/vae/` -- VAE module README and usage notes.
- `modules/cvae/` -- CVAE module README and usage notes.
- `models/AE/`, `models/VAE/`, `models/CVAE/` -- expected checkpoints are saved here (encoder/decoder state dicts).

## Modules

Each model has its own README with full explanations and figures :

- Autoencoder (AE): [modules/autoencoder/README.md](modules/autoencoder/README.md)
- Variational Autoencoder (VAE): [modules/vae/README.md](modules/vae/README.md)
- Conditional VAE (CVAE): [modules/cvae/README.md](modules/cvae/README.md)

## Usage instructions

1. Clone the repository

```bash
git clone git@github.com:JulienExr/Autoencoder-MNIST.git
(HTTPS : git clone https://github.com/JulienExr/Autoencoder-MNIST.git)
cd Autoencoder-MNIST
```

2. Create and activate a virtual environment :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Prepare data

MNIST will be downloaded automatically by `torchvision` into `./data` when you run training.

4. Run training (CLI)

`main.py` you can choose the model, dataset, and latent dimension:

You can visualize training outputs by adding the `--visualize` flag.

```bash
python main.py --model AE --dataset mnist --latent_dim 256
python main.py --model VAE --dataset mnist --latent_dim 32
```

Dataset options:
- `mnist` (default)
- `fashion_mnist` (more challenging, grayscale clothing items)

Example with Fashion-MNIST:

```bash
python main.py --model VAE --dataset fashion_mnist --latent_dim 128
```

4. Outputs

- Model checkpoints are saved under `models/AE/` and `models/VAE/` (encoder/decoder state dicts).
- Visual outputs are saved under `visu/<dataset>_<model>/` with subfolders `recon`, `pca`, `umap`, `interp`, and `noise`.
- If you want to try with your own model saved on models/* use : `streamlit run app.py`.