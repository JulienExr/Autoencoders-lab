import torch

from src.visualisation import Visualiser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_autoencoder(autoencoder, dataloader, test_loader, num_epochs=10, learning_rate=1e-3, latent_dim=256, device=device, visu_dir="mnist_autoencoder", visualise=True):
    autoencoder.to(device)
    autoencoder.train()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    losses = []
    visualiser = Visualiser(directory=visu_dir)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for (idx, data) in enumerate(dataloader):
            images, _ = data
            images = images.to(device)

            optimizer.zero_grad()
            decoded, encoded = autoencoder(images)
            loss = criterion(decoded, images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if (idx + 1) % 100 == 0 or idx == len(dataloader)-1:
                print(f"  Batch [{idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}", end='\r', flush=True)
        
        losses.append(epoch_loss / len(dataloader))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("\nGenerating visualizations...", visualise)
            if visualise:
                visualiser.visualize_reconstructions(autoencoder, test_loader, num_images=10, device=device, epoch=epoch+1)
                visualiser.pca_2d_latent(autoencoder, test_loader, device=device, epoch=epoch+1)
                visualiser.umap_2d_latent(autoencoder, test_loader, device=device, epoch=epoch+1)
                visualiser.interpolate_2_to_5(autoencoder, test_loader, device=device, epoch=epoch+1)
                visualiser.visu_from_noise(autoencoder, device=device, latent_dim=latent_dim, epoch=epoch+1, num_images=10)
            visualiser.plot_losses(losses)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.4f}")
    return losses

def train_vae(vae, dataloader, test_loader, num_epochs=10, learning_rate=1e-3, latent_dim=256, device=device, visu_dir="mnist_vae", visualise = True):
    vae.to(device)
    vae.train()

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    recon_losses = []
    kl_losses = []
    losses = []
    visualiser = Visualiser(directory=visu_dir, vae=True)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        for (idx, data) in enumerate(dataloader):
            images, _ = data
            images = images.to(device)

            decoded, mu, logvar = vae(images)
            recon_loss = criterion(decoded, images)
            kl_loss = -0.5 * torch.mean(1 + logvar -mu.pow(2) - logvar.exp())

            if epoch < 10:
                beta = 0.005
            else:
                beta = min(1.0, (epoch+1) / 500.0)
                beta *= 0.3
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            if (idx + 1) % 100 == 0 or idx == len(dataloader)-1:
                print(f"  Batch [{idx+1}/{len(dataloader)}], recon_Loss: {recon_loss.item():.4f} | KL_Loss: {kl_loss.item():.4f} | mu std: {mu.std().item():.4f} logvar mean: {logvar.mean().item():.4f}", end='\r', flush=True)

        losses.append(epoch_loss / len(dataloader))
        recon_losses.append(epoch_recon_loss / len(dataloader))
        kl_losses.append(epoch_kl_loss / len(dataloader))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("\nGenerating visualizations...", visualise)
            if visualise:
                visualiser.visualize_reconstructions(vae, test_loader, num_images=10, device=device, epoch=epoch+1)
                visualiser.pca_2d_latent(vae, test_loader, device=device, epoch=epoch+1)
                visualiser.umap_2d_latent(vae, test_loader, device=device, epoch=epoch+1)
                visualiser.interpolate_2_to_5(vae, test_loader, device=device, epoch=epoch+1)
                visualiser.visu_from_noise(vae, device=device, latent_dim=latent_dim, epoch=epoch+1, num_images=10)
            visualiser.plot_losses(losses)
            visualiser.plot_losses(recon_losses, name="reconstruction_loss")
            visualiser.plot_losses(kl_losses, name="kl_loss")
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.4f}, Recon Loss: {recon_losses[-1]:.4f}, KL Loss: {kl_losses[-1]:.4f}, Beta: {beta:.4f}")
    return losses

def train_cvae(cvae, dataloader, test_loader, num_epochs=10, learning_rate=1e-3, latent_dim=256, device=device, visu_dir="mnist_cvae", visualise = True):
    cvae.to(device)
    cvae.train()

    optimizer = torch.optim.Adam(cvae.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    recon_losses = []
    kl_losses = []
    losses = []
    def sample_labels(num_samples, device):
        return torch.randint(0, 10, (num_samples,), device=device, dtype=torch.long)

    visualiser = Visualiser(
        directory=visu_dir,
        vae=True,
        forward_fn=lambda model, images, labels: model(images, labels),
        label_sampler=sample_labels,
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        for (idx, data) in enumerate(dataloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            decoded, mu, logvar = cvae(images, labels)
            recon_loss = criterion(decoded, images)
            kl_loss = -0.5 * torch.mean(1 + logvar -mu.pow(2) - logvar.exp())

            if epoch < 10:
                beta = 0.005
            else:
                beta = min(1.0, (epoch+1) / 500.0)
                beta *= 0.3
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            if (idx + 1) % 100 == 0 or idx == len(dataloader)-1:
                print(f"  Batch [{idx+1}/{len(dataloader)}], recon_Loss: {recon_loss.item():.4f} | KL_Loss: {kl_loss.item():.4f} | mu std: {mu.std().item():.4f} logvar mean: {logvar.mean().item():.4f}", end='\r', flush=True)

        losses.append(epoch_loss / len(dataloader))
        recon_losses.append(epoch_recon_loss / len(dataloader))
        kl_losses.append(epoch_kl_loss / len(dataloader))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("\nGenerating visualizations...")
            if visualise:
                visualiser.visualize_reconstructions(cvae, test_loader, num_images=10, device=device, epoch=epoch+1)
                visualiser.pca_2d_latent(cvae, test_loader, device=device, epoch=epoch+1)
                visualiser.umap_2d_latent(cvae, test_loader, device=device, epoch=epoch+1)
                visualiser.interpolate_2_to_5(cvae, test_loader, device=device, epoch=epoch+1)
                visualiser.visu_from_noise_by_label(cvae, device=device, latent_dim=latent_dim, epoch=epoch+1, num_per_label=2)
            visualiser.plot_losses(losses)
            visualiser.plot_losses(recon_losses, name="reconstruction_loss")
            visualiser.plot_losses(kl_losses, name="kl_loss")
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.4f}, Recon Loss: {recon_losses[-1]:.4f}, KL Loss: {kl_losses[-1]:.4f}, Beta: {beta:.4f}")
    return losses