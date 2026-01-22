import torch
import streamlit as st
from pathlib import Path
import re
from torchvision.utils import make_grid

from src.ae import build_autoencoder
from src.vae import build_vae
from src.cvae import build_cvae
from src.data import get_mnist_dataloaders, get_fashion_mnist_dataloaders


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
        model = build_autoencoder(latent_dim=latent_dim)
        model_dir = Path("models/AE")
    elif model_type == "VAE":
        model = build_vae(latent_dim=latent_dim, mode="pp")
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


def get_dataloaders(dataset: str, normalize: bool):
    if dataset == "mnist":
        return get_mnist_dataloaders(batch_size=128, normalize=normalize)
    if dataset == "fashion_mnist":
        return get_fashion_mnist_dataloaders(batch_size=128, normalize=normalize)
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


st.title("AE / VAE / CVAE — Image Generator")

with st.sidebar:
    model_type = st.selectbox("Model", ["AE", "VAE", "CVAE"])
    dataset = st.selectbox("Dataset", ["mnist", "fashion_mnist"])
    if model_type == "AE":
        model_dir = Path("models/AE")
    elif model_type == "VAE":
        model_dir = Path("models/VAE")
    else:
        model_dir = Path("models/CVAE")

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

try:
    model, enc_path, dec_path = load_model(model_type, dataset, latent_dim, device)
    st.caption(f"Loaded: {enc_path} / {dec_path}")
except Exception as exc:
    st.error(str(exc))
    st.stop()

if model_type == "AE":
    st.subheader("Autoencoder")
    mode = st.radio("Generation mode", ["Reconstruction", "Random sampling"], horizontal=True)

    cache_key = ("AE", dataset, latent_dim, num_samples, mode)
    cached = st.session_state["cache"].get(cache_key)

    if mode == "Reconstruction":
        if cached is None:
            _, test_loader = get_dataloaders(dataset, normalize=False)
            batch = next(iter(test_loader))
            images = batch[0][:num_samples].to(device)
            with torch.no_grad():
                recon, _ = model(images)
            combined = torch.cat([images, recon], dim=0)
            st.session_state["cache"][cache_key] = combined.detach().cpu()
        combined = st.session_state["cache"][cache_key].to(device)

        st.write("Originals (top row) + Reconstructions (bottom row)")
        show_grid(combined, nrow=num_samples, normalize_range="0..1", display_width=display_width)

    else:
        if cached is None:
            z = torch.randn(num_samples, latent_dim, device=device)
            with torch.no_grad():
                samples = model.decoder(z)
            st.session_state["cache"][cache_key] = samples.detach().cpu()
        samples = st.session_state["cache"][cache_key].to(device)
        st.write("Random samples (AE decoder)")
        show_grid(samples, nrow=min(8, num_samples), normalize_range="0..1", display_width=display_width)

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
    show_grid(samples, nrow=min(8, num_samples), normalize_range="auto", display_width=display_width)

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
    show_grid(samples, nrow=min(8, num_samples), normalize_range="auto", display_width=display_width)
