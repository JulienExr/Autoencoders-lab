import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import umap


class Visualizer:

    def __init__(self, directory, model, forward_fn=None, label_sampler=None):
        self.directory = Path(directory)
        self.model = model
        self.forward_fn = forward_fn
        self.label_sampler = label_sampler
        self.base_dir = Path("visu") / self.directory
        for sub in ("recon", "pca", "umap", "interp", "noise", "prior"):
            (self.base_dir / sub).mkdir(parents=True, exist_ok=True)

    def _forward(self, model, images, labels=None):
        if self.forward_fn is not None:
            return self.forward_fn(model, images, labels)
        if labels is not None:
            try:
                return model(images, labels)
            except TypeError:
                return model(images)
        return model(images)

    def _encode(self, model, images, labels=None):
        if labels is not None:
            try:
                out = model.encoder(images, labels)
                if isinstance(out, tuple) and len(out) >= 2:
                    return out[0], out[1]
                return out
            except TypeError:
                return model.encoder(images)
        return model.encoder(images)

    def _decode(self, model, latents, labels=None):
        if labels is not None:
            if hasattr(model, "encoder") and hasattr(model.encoder, "embed"):
                embed_labels = model.encoder.embed(labels)
                return model.decoder(latents, embed_labels)
            try:
                return model.decoder(latents, labels)
            except TypeError:
                return model.decoder(latents)
        return model.decoder(latents)

    def _split_outputs(self, outputs):
        if self.model.__class__.__name__ == "VAE" or self.model.__class__.__name__ == "CVAE":
            decoded, mu, logvar = outputs
            return decoded, mu, logvar

        decoded, latent = outputs
        return decoded, latent, None

    def _reparameterize(self, mu, logvar):
        if logvar is None:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _prepare_image_for_imshow(self, image):
        image = image.detach().cpu().float()
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]

        img_min = image.min()
        img_max = image.max()
        if img_max > 1 or img_min < 0:
            image = (image - img_min) / (img_max - img_min + 1e-8)
        else:
            image = image.clamp(0, 1)

        return image

    def _imshow(self, image):
        prepared = self._prepare_image_for_imshow(image)
        if prepared.ndim == 2:
            plt.imshow(prepared, cmap="gray")
        else:
            plt.imshow(prepared)

    def visualize_reconstructions(self, model, dataloader, num_images=10, device='cuda', epoch=0):
        model.to(device)
        model.eval()

        images_shown = 0
        fig = plt.figure(figsize=(15, 4))

        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                decoded, _, _ = self._split_outputs(self._forward(model, images, labels))

                for i in range(images.size(0)):
                    if images_shown >= num_images:
                        break

                    plt.subplot(2, num_images, images_shown + 1)
                    self._imshow(images[i])
                    plt.axis('off')
                    if images_shown == 0:
                        fig.text(0.5,0.94,'Original Images', ha='center', va='bottom', fontsize=12, fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none')
                        )

                    plt.subplot(2, num_images, images_shown + 1 + num_images)
                    self._imshow(decoded[i])
                    plt.axis('off')
                    if images_shown == 0:
                        fig.text(0.5, 0.46, 'Reconstructed Images', ha='center', va='bottom', fontsize=12, fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none')
                        )
                    images_shown += 1
                if images_shown >= num_images:
                    break

        fig.subplots_adjust(hspace=0.6)
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
        fig.savefig(self.base_dir / "recon" / f"epoch_{epoch}.png")
        plt.close(fig)



    def pca_2d_latent(self, model, loader, device="cuda", max_batches=50, epoch=0):
        model.eval()
        Z_list, Y_list = [], []

        with torch.no_grad():
            for idx, (image, label) in enumerate(loader):
                if idx >= max_batches:
                    break
                image = image.to(device)
                label = label.to(device)
                out = self._forward(model, image, label)
                _, latent, logvar = self._split_outputs(out)
                encoded = latent

                Z_list.append(encoded.detach().cpu())
                Y_list.append(label.cpu())

        Z = torch.cat(Z_list, dim=0)
        Y = torch.cat(Y_list, dim=0)

        Z = Z.reshape(Z.size(0), -1)
        Z = Z - Z.mean(dim=0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Z, full_matrices=False)
        Z2 = Z @ Vh[:2].T

        fig = plt.figure(figsize=(7, 6))
        plt.scatter(Z2[:, 0].numpy(), Z2[:, 1].numpy(), s=8, c=Y.numpy())
        plt.colorbar()
        plt.title("PCA 2D of latent z")
        fig.savefig(self.base_dir / "pca" / f"epoch_{epoch}.png")
        plt.close(fig)

    def umap_2d_latent(self, model, loader, device="cuda", max_batches=50, epoch=0, n_neighbors=15, min_dist=0.1):
        model.eval()
        Z_list, Y_list = [], []

        with torch.no_grad():
            for idx, (image, label) in enumerate(loader):
                if idx >= max_batches:
                    break
                image = image.to(device)
                label = label.to(device)
                out = self._forward(model, image, label)
                _, latent, logvar = self._split_outputs(out)
                encoded = latent

                Z_list.append(encoded.detach().cpu())
                Y_list.append(label.cpu())

        Z = torch.cat(Z_list, dim=0)
        Y = torch.cat(Y_list, dim=0).numpy()

        Z = Z.reshape(Z.size(0), -1).numpy()

        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric="euclidean")
        Z2 = reducer.fit_transform(Z)

        fig = plt.figure(figsize=(7, 6))
        plt.scatter(Z2[:, 0], Z2[:, 1], s=8, c=Y)
        plt.colorbar()
        plt.title("UMAP 2D of latent z")
        fig.savefig(self.base_dir / "umap" / f"epoch_{epoch}.png")
        plt.close(fig)

    @torch.no_grad()
    def interpolate_2_to_5(self, model, loader, device, steps=10, epoch=0):
        model.eval()

        x2 = x5 = None

        for x, y in loader:
            for i in range(len(y)):
                if y[i] == 2 and x2 is None:
                    x2 = x[i]
                if y[i] == 5 and x5 is None:
                    x5 = x[i]
                if x2 is not None and x5 is not None:
                    break
            if x2 is not None and x5 is not None:
                break

        x2 = x2.unsqueeze(0).to(device)
        x5 = x5.unsqueeze(0).to(device)
        label2 = torch.tensor([2], device=device)
        label5 = torch.tensor([5], device=device)

        _, z2, logvar2 = self._split_outputs(self._forward(model, x2, label2))
        _, z5, logvar5 = self._split_outputs(self._forward(model, x5, label5))

        alphas = torch.linspace(0, 1, steps, device=device)
        alpha_shape = (steps,) + (1,) * (z2.dim() - 1)
        alphas = alphas.view(alpha_shape)
        z_interp = (1 - alphas) * z2 + alphas * z5

        interp_labels = torch.where(alphas < 0.5, label2.item(), label5.item())
        interp_labels = interp_labels.to(device=device, dtype=torch.long)

        x_interp = self._decode(model, z_interp, interp_labels).cpu()

        fig = plt.figure(figsize=(1.5 * steps, 2))
        for i in range(steps):
            plt.subplot(1, steps, i + 1)
            self._imshow(x_interp[i])
            plt.axis("off")
        plt.suptitle("Interpolation latent : 2 → 5")
        fig.savefig(self.base_dir / "interp" / f"epoch_{epoch}.png")
        plt.close(fig)

    @torch.no_grad()
    def visu_from_noise(self, model, device='cuda', latent_dim=256, epoch=0, num_images=10):
        """Sample random latents, decode them into images, then re-encode for diagnostics."""
        model.to(device)
        model.eval()

        if isinstance(latent_dim, (tuple, list)):
            z = torch.randn((num_images, *latent_dim), device=device)
        else:
            if latent_dim is None:
                raise ValueError("latent_dim must be provided for noise visualization")
            needs_spatial = not any(isinstance(m, torch.nn.Linear) for m in model.decoder.modules())
            if needs_spatial:
                z = torch.randn(num_images, int(latent_dim), 4, 4, device=device)
            else:
                z = torch.randn(num_images, int(latent_dim), device=device)
        labels = None
        if self.label_sampler is not None:
            labels = self.label_sampler(num_images, device)

        generated = self._decode(model, z, labels)
        if model.__class__.__name__ == "VAE" or model.__class__.__name__ == "CVAE":
            re_mu, _ = self._encode(model, generated, labels)
            reencoded = re_mu
        else:
            reencoded = self._encode(model, generated, labels)

        diff = reencoded - z
        latent_drift = diff.reshape(diff.size(0), -1).norm(dim=1).mean().item()

        save_dir = self.base_dir / "noise"
        fig = plt.figure(figsize=(1.5 * num_images, 2))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            self._imshow(generated[i])
            plt.axis('off')
        plt.suptitle("Samples from random latent")
        save_path = save_dir / f"epoch_{epoch}.png"
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

        return generated.detach().cpu()

    @torch.no_grad()
    def visu_from_noise_by_label(self, model, device='cuda', latent_dim=256, epoch=0, num_per_label=1, num_classes=10):
        """Generate random latents per label and decode them to visualize conditional sampling."""
        model.to(device)
        model.eval()

        total = num_classes * num_per_label
        z = torch.randn(total, latent_dim, device=device)
        labels = torch.arange(num_classes, device=device).repeat(num_per_label)

        generated = self._decode(model, z, labels)

        save_dir = self.base_dir / "noise"
        fig = plt.figure(figsize=(1.6 * num_classes, 1.6 * num_per_label))
        for i in range(total):
            row = i // num_classes
            col = i % num_classes
            plt.subplot(num_per_label, num_classes, i + 1)
            self._imshow(generated[i])
            plt.axis('off')
            if row == 0:
                plt.title(str(col), fontsize=10)

        plt.suptitle("Conditional samples by label")
        fig.tight_layout(rect=[0.05, 0.02, 1, 0.95])
        save_path = save_dir / f"epoch_{epoch}_by_label.png"
        fig.savefig(save_path)
        plt.close(fig)

        return generated.detach().cpu()

    def plot_losses(self, losses, name="loss"):
        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')        
        plt.savefig(self.base_dir / f"{name}.png")
        plt.close()

    @torch.no_grad()
    def visu_from_transformer_prior(
        self,
        transformer,
        vqvae,
        device='cuda',
        epoch=0,
        num_images=10,
        latent_shape=None,
        temperature=1.0,
        top_k=None,
    ):
        """Sample codebook indices with a Transformer prior, decode with VQ-VAE, and save a grid."""
        if latent_shape is None or len(latent_shape) != 3:
            raise ValueError("latent_shape must be provided as (C, H, W) for VQ-VAE decoding")

        transformer.to(device)
        transformer.eval()
        vqvae.to(device)
        vqvae.eval()

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
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                top_k = min(top_k, next_logits.size(-1))
                values, indices = torch.topk(next_logits, top_k, dim=-1)
                mask = torch.full_like(next_logits, float('-inf'))
                mask.scatter_(1, indices, values)
                next_logits = mask
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, next_token], dim=1)

        if use_sos:
            seq = seq[:, 1:]

        seq = seq[:, :latent_hw]

        embeddings = vqvae.vector_quantizer.embedding(seq)
        quantized = embeddings.view(num_images, h, w, -1).permute(0, 3, 1, 2).contiguous()

        generated = vqvae.decoder(quantized)

        save_dir = self.base_dir / "prior"
        fig = plt.figure(figsize=(1.5 * num_images, 2))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            self._imshow(generated[i])
            plt.axis('off')
        plt.suptitle("Transformer prior samples")
        fig.tight_layout()
        fig.savefig(save_dir / f"epoch_{epoch}.png")
        plt.close(fig)

        return generated.detach().cpu()