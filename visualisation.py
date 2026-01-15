import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import umap


class Visualiser:

    def __init__(self, directory, vae=False):
        self.directory = Path(directory)
        self.vae = vae
        self.base_dir = Path("visu") / self.directory
        for sub in ("recon", "pca", "umap", "interp", "noise"):
            (self.base_dir / sub).mkdir(parents=True, exist_ok=True)

    def _split_outputs(self, outputs):
        if self.vae:
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

    def visualize_reconstructions(self, model, dataloader, num_images=10, device='cuda', epoch=0):
        model.to(device)
        model.eval()

        images_shown = 0
        fig = plt.figure(figsize=(15, 4))

        with torch.no_grad():
            for data in dataloader:
                images, _ = data
                images = images.to(device)

                decoded, _, _ = self._split_outputs(model(images))

                for i in range(images.size(0)):
                    if images_shown >= num_images:
                        break

                    plt.subplot(2, num_images, images_shown + 1)
                    plt.imshow(images[i].cpu().squeeze(), cmap='gray')
                    plt.axis('off')
                    if images_shown == 0:
                        fig.text(0.5,0.94,'Original Images', ha='center', va='bottom', fontsize=12, fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none')
                        )

                    plt.subplot(2, num_images, images_shown + 1 + num_images)
                    plt.imshow(decoded[i].cpu().squeeze(), cmap='gray')
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
                out = model(image)
                _, latent, logvar = self._split_outputs(out)
                encoded = latent

                Z_list.append(encoded.detach().cpu())
                Y_list.append(label.cpu())

        Z = torch.cat(Z_list, dim=0)
        Y = torch.cat(Y_list, dim=0)

        Z = Z - Z.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
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
                out = model(image)
                _, latent, logvar = self._split_outputs(out)
                encoded = latent

                Z_list.append(encoded.detach().cpu())
                Y_list.append(label.cpu())

        Z = torch.cat(Z_list, dim=0).numpy()
        Y = torch.cat(Y_list, dim=0).numpy()

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

        _, z2, logvar2 = self._split_outputs(model(x2))
        _, z5, logvar5 = self._split_outputs(model(x5))

        alphas = torch.linspace(0, 1, steps, device=device)
        z_interp = (1 - alphas[:, None]) * z2 + alphas[:, None] * z5

        x_interp = model.decoder(z_interp).cpu()

        fig = plt.figure(figsize=(1.5 * steps, 2))
        for i in range(steps):
            plt.subplot(1, steps, i + 1)
            plt.imshow(x_interp[i, 0], cmap="gray")
            plt.axis("off")
        plt.suptitle("Interpolation latent : 2 → 5")
        fig.savefig(self.base_dir / "interp" / f"epoch_{epoch}.png")
        plt.close(fig)

    @torch.no_grad()
    def visu_from_noise(self, model, device='cuda', latent_dim=256, epoch=0, num_images=10):
        """Sample random latents, decode them into images, then re-encode for diagnostics."""
        model.to(device)
        model.eval()

        z = torch.randn(num_images, latent_dim, device=device)
        generated = model.decoder(z)
        if self.vae:
            re_mu, _ = model.encoder(generated)
            reencoded = re_mu
        else:
            reencoded = model.encoder(generated)

        latent_drift = torch.norm(reencoded - z, dim=1).mean().item()

        save_dir = self.base_dir / "noise"
        fig = plt.figure(figsize=(1.5 * num_images, 2))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(generated[i, 0].detach().cpu(), cmap='gray')
            plt.axis('off')
        plt.suptitle("Samples from random latent")
        save_path = save_dir / f"epoch_{epoch}.png"
        fig.tight_layout()
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