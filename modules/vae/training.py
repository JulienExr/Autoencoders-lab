import torch

from src.visualization import Visualizer
from src.loss import VGGPerceptualLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_vae(vae, dataloader, test_loader, num_epochs=10, learning_rate=1e-3, latent_dim=256, device=device, visu_dir="mnist_vae", visualize = True):
    vae.to(device)
    vae.train()

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = torch.nn.MSELoss()
    perceptual_criterion = VGGPerceptualLoss().to(device)

    recon_losses = []
    kl_losses = []
    perceptual_losses = []
    losses = []
    visualizer = Visualizer(directory=visu_dir, model=vae)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_perceptual_loss = 0.0
        for (idx, data) in enumerate(dataloader):
            images, _ = data
            images = images.to(device)

            decoded, mu, logvar = vae(images)
            recon_loss = criterion(decoded, images)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            vgg_loss = perceptual_criterion(decoded, images)
            
            if epoch < 10:
                beta = 0.02
                beta_vgg = 0.02
            else:
                beta = min(0.07, (epoch + 1) / 100.0)
                beta_vgg = min(0.02, (epoch + 1) / 100.0)
            

            loss = recon_loss + beta * kl_loss + beta_vgg * vgg_loss

            if not torch.isfinite(loss):
                print(f"\nSkipping non-finite loss at epoch {epoch+1}, batch {idx+1}")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_perceptual_loss += vgg_loss.item()


            if (idx + 1) % 100 == 0 or idx == len(dataloader)-1:
                print(f"  Batch [{idx+1}/{len(dataloader)}], recon_Loss: {recon_loss.item():.4f} | KL_Loss: {kl_loss.item():.4f} | mu std: {mu.std().item():.4f} logvar mean: {logvar.mean().item():.4f}", end='\r', flush=True)

        losses.append(epoch_loss / len(dataloader))
        recon_losses.append(epoch_recon_loss / len(dataloader))
        kl_losses.append(epoch_kl_loss / len(dataloader))
        perceptual_losses.append(epoch_perceptual_loss / len(dataloader))
        scheduler.step(losses[-1])
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("\nGenerating visualizations...", visualize)
            if visualize:
                visualizer.visualize_reconstructions(vae, test_loader, num_images=10, device=device, epoch=epoch+1)
                # visualizer.pca_2d_latent(vae, test_loader, device=device, epoch=epoch+1)
                # visualizer.umap_2d_latent(vae, test_loader, device=device, epoch=epoch+1)
                visualizer.interpolate_2_to_5(vae, test_loader, device=device, epoch=epoch+1)
                visualizer.visu_from_noise(vae, device=device, latent_dim=latent_dim, epoch=epoch+1, num_images=10)
            visualizer.plot_losses(losses)
            visualizer.plot_losses(recon_losses, name="reconstruction_loss")
            visualizer.plot_losses(kl_losses, name="kl_loss")
            visualizer.plot_losses(perceptual_losses, name="perceptual_loss")
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.4f}, Recon Loss: {recon_losses[-1]:.4f}, KL Loss: {kl_losses[-1]:.4f}, Perceptual Loss: {perceptual_losses[-1]:.4f}, Beta: {beta:.4f}, Beta VGG: {beta_vgg:.4f}")
        print(f"poids des losses :" f" recon={recon_losses[-1]/losses[-1] * 100:.4f} %, kl={kl_losses[-1] * beta/losses[-1] * 100:.4f} % , perceptual={perceptual_losses[-1] * beta_vgg/losses[-1] * 100:.4f} %")
    return losses