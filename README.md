# Autoencoder & VAE on MNIST

This repository contains simple PyTorch implementations of an Autoencoder (AE) and a Variational Autoencoder (VAE) trained on MNIST. The code is minimal and focused on experiments: training, latent-space inspection, reconstructions, interpolation (2 → 5), and sampling from the latent space.

## Quick project layout

- `ae.py` -- AE model (encoder + decoder).
- `vae.py` -- VAE model (probabilistic encoder producing μ and logvar + decoder). Two encoder/decoder variants exist (`default` and `pp`).
- `training.py` -- training loops for AE and VAE. Visualization is called automatically from here each epoch via `Visualiser`.
- `main.py` -- example entry points. By default it runs AE training then VAE training (see note below).
- `visu.py` -- `Visualiser` helper used by `training.py` to save reconstructions, PCA plots, interpolations and noise samples into `visu/`.
- `data.py` -- MNIST dataloader helpers.
- `model/AE/`, `model/VAE/` -- expected checkpoints are saved here (encoder/decoder state dicts).


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

The VAE predicts μ and logvar for each input and uses the reparameterization trick to sample z. The loss combines reconstruction + KL divergence to a prior (N(0,I)). This changes behavior:

- The latent space is pushed toward the prior, which makes sampling from N(0,I) meaningful.
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

## Usage instructions

1. Create and activate a virtual environment :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data

MNIST will be downloaded automatically by `torchvision` into `./data` when you run training.

3. Run training

```bash
python main.py
```

- Default behavior: `main.py` triggers AE training first, then VAE training.
- To run only one model, edit `main.py` and comment out the unwanted call (for example comment `main_VAE()` to train only the AE).

4. Outputs

- Model checkpoints are saved under `model/AE/` and `model/VAE/` (encoder/decoder state dicts).
- Visual outputs are saved under `visu/<model_name>/` with subfolders `recon`, `pca`, `umap`, `interp`, and `noise`.

## Small tips

- AE latent dim defaults to 256; `main.py` calls VAE with latent_dim=32 and `mode="pp"` (a slightly deeper variant).
- The VAE training schedule in `training.py` uses a small beta early on and ramps it; tweak that schedule if you want sharper reconstructions vs tighter latent.