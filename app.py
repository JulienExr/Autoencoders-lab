import torch
import streamlit as st
from pathlib import Path
import re
from torchvision.utils import make_grid

from modules.autoencoder.ae import build_autoencoder, build_autoencoder_cifar
from modules.vae.vae import build_vae, build_vae_cifar
from modules.cvae.cvae import build_cvae
from modules.vq_vae.vq_vae import build_vqvae
from modules.vq_vae.tranformer_prior import build_transformer_prior
from src.data import get_mnist_dataloaders, get_fashion_mnist_dataloaders, get_cifar10_dataloaders


st.set_page_config(page_title="AE / VAE / CVAE Generator", layout="wide")


def resolve_checkpoint_paths(model_dir: Path, dataset: str, latent_dim: int | None = None):
    candidates = [
        (f"encoder_{dataset}_{latent_dim}.pth", f"decoder_{dataset}_{latent_dim}.pth")
        if latent_dim is not None
        else None,
        (f"encoder_{dataset}.pth", f"decoder_{dataset}.pth"),
        ("encoder.pth", "decoder.pth"),
        ("encoder_mnist.pth", "decoder_mnist.pth"),
        ("encoder_fashion_mnist.pth", "decoder_fashion_mnist.pth"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        enc_name, dec_name = candidate
        enc_path = model_dir / enc_name
        dec_path = model_dir / dec_name
        if enc_path.exists() and dec_path.exists():
            return enc_path, dec_path
    return None, None


def resolve_vqvae_paths(model_dir: Path, dataset: str):
    enc_path = model_dir / f"encoder_{dataset}.pth"
    dec_path = model_dir / f"decoder_{dataset}.pth"
    vq_path = model_dir / f"vq_{dataset}.pth"
    if enc_path.exists() and dec_path.exists() and vq_path.exists():
        return enc_path, dec_path, vq_path
    return None, None, None


def get_available_latent_dims(model_dir: Path, dataset: str) -> list[int]:
    pattern = re.compile(rf"^encoder_{re.escape(dataset)}_(\d+)\.pth$")
    dims: set[int] = set()
    for path in model_dir.glob(f"encoder_{dataset}_*.pth"):
        match = pattern.match(path.name)
        if match:
            dims.add(int(match.group(1)))
    return sorted(dims)


@st.cache_resource
def load_model(model_type: str, dataset: str, latent_dim: int, device: str):
    if model_type == "AE":
        model = build_autoencoder_cifar(latent_dim=latent_dim) if dataset == "cifar10" else build_autoencoder(latent_dim=latent_dim)
        model_dir = Path("models/AE")
    elif model_type == "VAE":
        model = build_vae_cifar(latent_dim=latent_dim) if dataset == "cifar10" else build_vae(latent_dim=latent_dim, mode="pp")
        model_dir = Path("models/VAE")
    elif model_type == "CVAE":
        model = build_cvae(latent_dim=latent_dim)
        model_dir = Path("models/CVAE")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    enc_path, dec_path = resolve_checkpoint_paths(model_dir, dataset, latent_dim=latent_dim)
    if enc_path is None or dec_path is None:
        raise FileNotFoundError(
            f"No checkpoints found in {model_dir} for dataset '{dataset}'."
        )

    model.encoder.load_state_dict(torch.load(enc_path, map_location=device))
    model.decoder.load_state_dict(torch.load(dec_path, map_location=device))
    model.to(device)
    model.eval()
    return model, enc_path, dec_path


@st.cache_resource
def load_vqvae(dataset: str, device: str):
    model_dir = Path("models/VQ-VAE")
    enc_path, dec_path, vq_path = resolve_vqvae_paths(model_dir, dataset)
    if enc_path is None or dec_path is None or vq_path is None:
        raise FileNotFoundError(
            f"No VQ-VAE checkpoints found in {model_dir} for dataset '{dataset}'."
        )

    vq_state = torch.load(vq_path, map_location=device)
    embedding_weight = vq_state.get("embedding.weight")
    if embedding_weight is None:
        raise ValueError("Invalid VQ-VAE codebook state: missing embedding.weight.")

    num_embeddings, embedding_dim = embedding_weight.shape
    vqvae = build_vqvae(embedding_dim=embedding_dim, num_embeddings=num_embeddings, commitment_cost=0.25)
    vqvae.encoder.load_state_dict(torch.load(enc_path, map_location=device))
    vqvae.decoder.load_state_dict(torch.load(dec_path, map_location=device))
    vqvae.vector_quantizer.load_state_dict(vq_state)
    vqvae.to(device)
    vqvae.eval()
    return vqvae, enc_path, dec_path, vq_path


@st.cache_resource
def load_transformer_prior(dataset: str, device: str):
    model_dir = Path("models/TransformerPrior")
    transformer_path = model_dir / f"transformer_{dataset}.pth"
    if not transformer_path.exists():
        raise FileNotFoundError(
            f"No Transformer prior checkpoint found in {model_dir} for dataset '{dataset}'."
        )

    transformer_state = torch.load(transformer_path, map_location=device)
    vocab_size = transformer_state["token_embedding.weight"].shape[0]
    embedding_dim = transformer_state["token_embedding.weight"].shape[1]
    seq_len = transformer_state["position_embedding.weight"].shape[0]

    transformer = build_transformer_prior(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_head=4,
        num_layers=6,
        seq_len=seq_len,
    )
    transformer.load_state_dict(transformer_state, strict=True)
    transformer.to(device)
    transformer.eval()
    return transformer, transformer_path


def get_dataloaders(dataset: str, normalize: bool):
    if dataset == "mnist":
        return get_mnist_dataloaders(batch_size=128, normalize=normalize)
    if dataset == "fashion_mnist":
        return get_fashion_mnist_dataloaders(batch_size=128, normalize=normalize)
    if dataset == "cifar10":
        return get_cifar10_dataloaders(batch_size=128, normalize=normalize)
    raise ValueError(f"Unsupported dataset: {dataset}")


def to_display(imgs: torch.Tensor, normalize_range: str):
    if normalize_range == "-1..1":
        if imgs.min().item() < 0:
            imgs = (imgs + 1) / 2
        imgs = imgs.clamp(0, 1)
        return imgs
    if normalize_range == "auto":
        if imgs.ndim == 3:
            imgs = imgs.unsqueeze(0)
        min_vals = imgs.amin(dim=(1, 2, 3), keepdim=True)
        max_vals = imgs.amax(dim=(1, 2, 3), keepdim=True)
        denom = (max_vals - min_vals).clamp_min(1e-6)
        imgs = (imgs - min_vals) / denom
        return imgs.clamp(0, 1)
    imgs = imgs.clamp(0, 1)
    return imgs


def show_grid(imgs: torch.Tensor, nrow: int, normalize_range: str, display_width: int):
    imgs = to_display(imgs, normalize_range)
    grid = make_grid(imgs, nrow=nrow, padding=2)
    st.image(grid.permute(1, 2, 0).cpu().numpy(), width=display_width)


@torch.no_grad()
def sample_codebook_indices(
    transformer,
    vqvae,
    latent_shape,
    num_images=10,
    temperature=1.0,
    top_k=None,
    device="cuda",
):
    if latent_shape is None or len(latent_shape) != 3:
        raise ValueError("latent_shape must be provided as (C, H, W)")

    _, h, w = latent_shape
    latent_hw = h * w

    num_embeddings = vqvae.vector_quantizer.num_embeddings
    vocab_size = transformer.token_embedding.num_embeddings
    max_seq_len = transformer.position_embedding.num_embeddings

    use_sos = vocab_size == (num_embeddings + 1) and max_seq_len == (latent_hw + 1)
    if max_seq_len < latent_hw:
        raise ValueError(
            f"Transformer seq_len ({max_seq_len}) is smaller than latent grid ({latent_hw})."
        )

    if use_sos:
        seq = torch.full((num_images, 1), num_embeddings, device=device, dtype=torch.long)
        total_len = latent_hw + 1
    else:
        seq = torch.randint(0, num_embeddings, (num_images, 1), device=device, dtype=torch.long)
        total_len = latent_hw

    for _ in range(seq.size(1), total_len):
        logits = transformer(seq)
        next_logits = logits[:, -1, :] / max(float(temperature), 1e-6)
        if top_k is not None and top_k > 0:
            k = min(int(top_k), next_logits.size(-1))
            values, indices = torch.topk(next_logits, k, dim=-1)
            masked = torch.full_like(next_logits, float("-inf"))
            masked.scatter_(1, indices, values)
            next_logits = masked
        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_token], dim=1)

    if use_sos:
        seq = seq[:, 1:]

    return seq[:, :latent_hw]


@torch.no_grad()
def generate_images_with_transformer(
    transformer,
    vqvae,
    latent_shape,
    num_images=10,
    temperature=1.0,
    top_k=None,
    device="cuda",
):
    indices = sample_codebook_indices(
        transformer,
        vqvae,
        latent_shape=latent_shape,
        num_images=num_images,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )

    _, h, w = latent_shape
    embeddings = vqvae.vector_quantizer.embedding(indices)
    quantized = embeddings.view(num_images, h, w, -1).permute(0, 3, 1, 2).contiguous()
    images = vqvae.decoder(quantized)
    return images.detach().cpu()


st.title("AE / VAE / CVAE / VQ-VAE — Image Generator")

with st.sidebar:
    model_type = st.selectbox("Model", ["AE", "VAE", "CVAE", "VQ-VAE"])
    if model_type in {"AE", "VAE"}:
        dataset_options = ["mnist", "fashion_mnist", "cifar10"]
    elif model_type == "CVAE":
        dataset_options = ["mnist", "fashion_mnist"]
    else:
        dataset_options = ["mnist", "fashion_mnist"]
    dataset = st.selectbox("Dataset", dataset_options)
    if model_type == "AE":
        model_dir = Path("models/AE")
    elif model_type == "VAE":
        model_dir = Path("models/VAE")
    elif model_type == "CVAE":
        model_dir = Path("models/CVAE")
    else:
        model_dir = Path("models/VQ-VAE")

    if model_type == "VQ-VAE":
        latent_dim = 0
    else:
        available_dims = get_available_latent_dims(model_dir, dataset)
        if available_dims:
            latent_dim = st.selectbox("Latent dimension", available_dims)
        else:
            latent_dim = st.slider("Latent dimension", min_value=8, max_value=512, value=32, step=8)
    num_samples = st.slider("Number of images", min_value=1, max_value=32, value=16, step=1)
    display_width = st.slider("Display width (px)", min_value=400, max_value=1600, value=900, step=50)

if "cache" not in st.session_state:
    st.session_state["cache"] = {}


device = "cuda" if torch.cuda.is_available() else "cpu"

normalize_inputs = (model_type != "AE") or dataset == "cifar10"

try:
    if model_type == "VQ-VAE":
        model, enc_path, dec_path, vq_path = load_vqvae(dataset, device)
        st.caption(f"Loaded: {enc_path} / {dec_path} / {vq_path}")
    else:
        model, enc_path, dec_path = load_model(model_type, dataset, latent_dim, device)
        st.caption(f"Loaded: {enc_path} / {dec_path}")
except Exception as exc:
    st.error(str(exc))
    st.stop()

if model_type == "AE":
    st.subheader("Autoencoder")
    mode = st.radio("Generation mode", ["Reconstruction", "Random sampling"], horizontal=True)

    display_range = "-1..1" if dataset == "cifar10" else "0..1"

    cache_key = ("AE", dataset, latent_dim, num_samples, mode)
    cached = st.session_state["cache"].get(cache_key)

    if mode == "Reconstruction":
        if cached is None:
            _, test_loader = get_dataloaders(dataset, normalize=normalize_inputs)
            batch = next(iter(test_loader))
            images = batch[0][:num_samples].to(device)
            with torch.no_grad():
                recon, _ = model(images)
            combined = torch.cat([images, recon], dim=0)
            st.session_state["cache"][cache_key] = combined.detach().cpu()
        combined = st.session_state["cache"][cache_key].to(device)

        st.write("Originals (top row) + Reconstructions (bottom row)")
        show_grid(combined, nrow=num_samples, normalize_range=display_range, display_width=display_width)

    else:
        if cached is None:
            if dataset == "cifar10":
                z = torch.randn(num_samples, latent_dim, 8, 8, device=device)
            else:
                z = torch.randn(num_samples, latent_dim, device=device)
            with torch.no_grad():
                samples = model.decoder(z)
            st.session_state["cache"][cache_key] = samples.detach().cpu()
        samples = st.session_state["cache"][cache_key].to(device)
        st.write("Random samples (AE decoder)")
        show_grid(samples, nrow=min(8, num_samples), normalize_range=display_range, display_width=display_width)

elif model_type == "VAE":
    st.subheader("Variational Autoencoder")

    cache_key = ("VAE", dataset, latent_dim, num_samples)
    cached = st.session_state["cache"].get(cache_key)
    if cached is None:
        z = torch.randn(num_samples, latent_dim, device=device)
        with torch.no_grad():
            samples = model.decoder(z)
        st.session_state["cache"][cache_key] = samples.detach().cpu()
    samples = st.session_state["cache"][cache_key].to(device)

    st.write("Samples from $z \sim \mathcal{N}(0, I)$")
    show_grid(samples, nrow=min(8, num_samples), normalize_range="-1..1", display_width=display_width)

elif model_type == "CVAE":
    st.subheader("Conditional VAE")
    label_mode = st.radio("Label selection", ["Fixed label", "Random labels"], horizontal=True)
    fixed_label = st.selectbox("Digit label", list(range(10))) if label_mode == "Fixed label" else 0

    if label_mode == "Fixed label":
        labels = torch.full((num_samples,), int(fixed_label), dtype=torch.long, device=device)
    else:
        labels = torch.randint(0, 10, (num_samples,), device=device)

    cache_key = ("CVAE", dataset, latent_dim, num_samples, label_mode, int(fixed_label))
    cached = st.session_state["cache"].get(cache_key)
    if cached is None:
        z = torch.randn(num_samples, latent_dim, device=device)
        with torch.no_grad():
            embed_label = model.encoder.embed(labels)
            samples = model.decoder(z, embed_label)
        st.session_state["cache"][cache_key] = samples.detach().cpu()
    samples = st.session_state["cache"][cache_key].to(device)

    st.write("Samples from $z \sim \mathcal{N}(0, I)$ with label conditioning")
    show_grid(samples, nrow=min(8, num_samples), normalize_range="-1..1", display_width=display_width)

elif model_type == "VQ-VAE":
    st.subheader("VQ-VAE + Transformer Prior")
    mode = st.radio("Generation mode", ["Reconstruction", "Random codebook", "Transformer prior"], horizontal=True)
    temperature = st.slider("Temperature", min_value=0.2, max_value=2.0, value=1.0, step=0.1)
    top_k = st.slider("Top-k (0 = disabled)", min_value=0, max_value=512, value=128, step=16)

    cache_key = ("VQ-VAE", dataset, num_samples, mode, float(temperature), int(top_k))
    cached = st.session_state["cache"].get(cache_key)

    if cached is None:
        if mode == "Reconstruction":
            _, test_loader = get_dataloaders(dataset, normalize=normalize_inputs)
            batch = next(iter(test_loader))
            images = batch[0][:num_samples].to(device)
            with torch.no_grad():
                recon, _ = model(images)
            combined = torch.cat([images, recon], dim=0)
            st.session_state["cache"][cache_key] = combined.detach().cpu()
        else:
            _, test_loader = get_dataloaders(dataset, normalize=normalize_inputs)
            batch = next(iter(test_loader))
            sample = batch[0][:1].to(device)
            with torch.no_grad():
                encoded = model.encoder(sample)
            latent_shape = tuple(encoded.shape[1:])

            if mode == "Random codebook":
                _, h, w = latent_shape
                random_indices = torch.randint(
                    0,
                    model.vector_quantizer.num_embeddings,
                    (num_samples, h * w),
                    device=device,
                    dtype=torch.long,
                )
                embeddings = model.vector_quantizer.embedding(random_indices)
                quantized = embeddings.view(num_samples, h, w, -1).permute(0, 3, 1, 2).contiguous()
                images = model.decoder(quantized)
                st.session_state["cache"][cache_key] = images.detach().cpu()
            else:
                transformer, transformer_path = load_transformer_prior(dataset, device)
                samples = generate_images_with_transformer(
                    transformer,
                    model,
                    latent_shape=latent_shape,
                    num_images=num_samples,
                    temperature=temperature,
                    top_k=None if top_k == 0 else top_k,
                    device=device,
                )
                st.session_state["cache"][cache_key] = samples.detach().cpu()

    results = st.session_state["cache"][cache_key].to(device)

    if mode == "Reconstruction":
        st.write("Originals (top row) + Reconstructions (bottom row)")
        show_grid(results, nrow=num_samples, normalize_range="-1..1", display_width=display_width)
    elif mode == "Random codebook":
        st.write("Random codebook samples (VQ-VAE decoder)")
        show_grid(results, nrow=min(8, num_samples), normalize_range="-1..1", display_width=display_width)
    else:
        st.write("Samples from Transformer prior")
        show_grid(results, nrow=min(8, num_samples), normalize_range="-1..1", display_width=display_width)
