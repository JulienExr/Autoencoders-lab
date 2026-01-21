# Autoencoder & VAE on MNIST

## Overview

This repository contains clean PyTorch implementations of an Autoencoder (AE) and Variational Autoencoder (VAE) trained on MNIST.

**Try the interactive demo:** https://julienexr-autoencooder-mnist-app-sehuso.streamlit.app/

## Table of contents

- [Quick project layout](#quick-project-layout)
- [Autoencoder (AE) -- summary and visualizations](#autoencoder-ae----summary-and-visualizations)
- [Variational Autoencoder (VAE) -- summary and visualizations](#variational-autoencoder-vae----summary-and-visualizations)
- [Conditional VAE (CVAE) -- summary and visualizations](#conditional-vae-cvae----summary-and-visualizations)
- [Fashion-MNIST example renders](#fashion-mnist-example-renders)
- [Usage instructions](#usage-instructions)
- [Small tips](#small-tips)

## Quick project layout

- `ae.py` -- AE model (encoder + decoder).
- `vae.py` -- VAE model (probabilistic encoder producing μ and logvar + decoder). Two encoder/decoder variants exist (`default` and `pp`).
- `cvae.py` -- CVAE model (VAE conditioned on class labels).
- `training.py` -- training loops for AE and VAE. Visualization is called automatically from here each epoch via `Visualiser`.
- `main.py` -- example entry points. By default it runs AE training then VAE training (see note below).
- `visualisation.py` -- `Visualiser` helper used by `training.py` to save reconstructions, PCA plots, interpolations and noise samples into `visu/`.
- `data.py` -- MNIST dataloader helpers.
- `model/AE/`, `model/VAE/`, `model/CVAE/` -- expected checkpoints are saved here (encoder/decoder state dicts).


## Autoencoder (AE) -- summary and visualizations

The AE is a deterministic encoder/decoder pair trained to minimize reconstruction error (MSE in `training.py`).

- Reconstructions - grid showing original vs reconstructed images.
- Latent PCA -- project the encoded vectors to 2D with PCA and plot colored points by digit label.
- Latent UMAP -- non-linear 2D projection of the latent space (often clearer than PCA).
- Interpolation (2 → 5) -- linear interpolation in latent space between an example "2" and an example "5".
- Sampling from random latent vectors -- decode Gaussian random vectors and inspect outputs.

Example :

<figure style="text-align: center;">
  <img src="demo/recon_ae_20.png" alt="AE reconstruction example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Original images (top row) and their reconstructions (bottom row).</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="demo/inter_ae_20.png" alt="AE interpolation 2 to 5 example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Decoded images along a linear path in latent space between a sampled "2" and a sampled "5".</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="demo/pca_ae_20.png" alt="AE PCA example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: PCA 2D projection of AE latent vectors, colored by digit label.</figcaption>
</figure>

- The PCA 2D scatter shows how encoded vectors cluster by digit label. We can observe compact clusters but also empty regions: AE latent space is not forced to follow a known prior, so regions between clusters can be meaningless.

<figure style="text-align: center;">
  <img src="demo/umap_ae_20.png" alt="AE UMAP example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: UMAP 2D projection of AE latent vectors.</figcaption>
</figure>

- With UMAP, we can see a real separation between digit clusters, but also some curved manifolds and local neighborhoods that PCA may not reveal as clearly.

<figure style="text-align: center;">
  <img src="demo/noise_ae_20.png" alt="AE noise sampling example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Random Gaussian latents decoded by the AE.</figcaption>
</figure>

Sampling / noise generation (AE)
- The AE's decoder is trained to reconstruct images from encoded latents produced by its encoder. It is not trained to decode vectors sampled from a standard Gaussian prior. As a result, decoding random Gaussian noise often yields garbage or highly distorted digits.

Why AE sampling fails :
- No regularization: the AE encoder can place encoded points anywhere in latent space; there is no force to match a Gaussian prior.
- Decoder overfits to encoder manifold: decoder learns to map encoder outputs back to images, but random z are off-manifold.
- Conclusion: A plain AE is good for compression and reconstruction, but not a reliable generative model by sampling random latents.

## Variational Autoencoder (VAE) -- summary and visualizations

The VAE predicts μ and logvar for each input and uses the reparameterization trick to sample z. The loss combines reconstruction + KL divergence to a prior (N(0,1)). This changes behavior:

- The latent space is pushed toward the prior, which makes sampling from N(0,1) meaningful.
- Interpolations tend to be smoother and decoded samples from the prior are more coherent than for a plain AE.

Example placeholders:

<figure style="text-align: center;">
  <img src="demo/recon_vae_50.png" alt="VAE reconstruction example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Originals vs VAE reconstructions. VAE reconstructions can be slightly blurrier depending on KL weight.</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="demo/pca_vae_50.png" alt="VAE PCA example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: PCA 2D projection of VAE latent vectors, colored by digit label.</figcaption>
</figure>

- VAE PCA shows a more continuous embedding: latent vectors are more evenly distributed following the Gaussian prior.

<figure style="text-align: center;">
  <img src="demo/umap_vae_50.png" alt="VAE UMAP example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: UMAP 2D projection of VAE latent vectors.</figcaption>
</figure>

- In the VAE case, UMAP show a smoother, more continuous manifold than the AE, thanks to KL regularization.

<figure style="text-align: center;">
  <img src="demo/noise_vae_50.png" alt="VAE sampled images example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Images decoded from samples drawn from N(0,I) in the VAE latent space.</figcaption>
</figure>

- These samples should look more digit-like than AE noise samples because the VAE latent is regularized toward the Gaussian prior.

## Conditional VAE (CVAE) -- summary and visualizations

The CVAE extends the VAE by conditioning both encoder and decoder on the class label $y$. This lets you guide generation by specifying a digit class.

- Conditioning: labels are injected into the encoder/decoder (e.g., via concatenation or embeddings) so the latent is informed by the target class.
- Controlled sampling: you can sample $z \sim \mathcal{N}(0, 1)$ and decode with a chosen label to generate class-specific digits.
- Reconstructions: expect reconstructions similar to VAE, but with better control over class identity.

Example placeholders:

<figure style="text-align: center;">
  <img src="demo/recon_cvae_50.png" alt="CVAE reconstruction example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Originals vs CVAE reconstructions conditioned on labels.</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="demo/pca_cvae_50.png" alt="CVAE PCA example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: PCA 2D projection of CVAE latent vectors, colored by digit label.</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="demo/noise_cvae_50.png" alt="CVAE sampled images example" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Samples from N(0, 1) decoded with a fixed label (class-conditional generation).</figcaption>
</figure>

- A CVAE is useful when you need controllable generation (choose a digit) while still benefiting from a structured latent prior. The trade-off is that conditioning slightly constrains the latent space, so diversity within a class can depend on how well labels are injected.

## Fashion-MNIST example renders

Below are example placeholders from the Fashion-MNIST dataset. This dataset is more complex than MNIST, so reconstructions and latent embeddings can look less clean. 

<figure style="text-align: center;">
  <img src="demo/fashion_recon_ae_20.png" alt="Fashion-MNIST AE reconstructions" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Fashion-MNIST AE reconstructions after 20 epochs.</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="demo/fashion_pca_vae_50.png" alt="Fashion-MNIST VAE PCA" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Fashion-MNIST VAE PCA after 50 epochs.</figcaption>
</figure>

<figure style="text-align: center;">
  <img src="demo/fashion_noise_vae_50.png" alt="Fashion-MNIST VAE sampling" style="max-width: 100%;" />
  <figcaption style="font-style: italic;">Figure: Fashion-MNIST VAE samples drawn from the latent prior.</figcaption>
</figure>

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

You can visualize training outputs by adding the `--visualise` flag.

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

- Model checkpoints are saved under `model/AE/` and `model/VAE/` (encoder/decoder state dicts).
- Visual outputs are saved under `visu/<dataset>_<model>/` with subfolders `recon`, `pca`, `umap`, `interp`, and `noise`.
- If you want to try with your own model saved on model/* use : `streamlit run app.py`.
## Small tips

- The VAE training schedule in `training.py` uses a small beta early on and ramps it; tweak that schedule if you want sharper reconstructions vs tighter latent.