import torch

from src.visualization import Visualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_autoencoder(autoencoder, dataloader, test_loader, num_epochs=10, learning_rate=1e-3, latent_dim=256, device=device, visu_dir="mnist_autoencoder", visualize=True):
    autoencoder.to(device)
    autoencoder.train()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    losses = []
    visualizer = Visualizer(directory=visu_dir, model= autoencoder)

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
            print("\nGenerating visualizations...", visualize)
            if visualize:
                visualizer.visualize_reconstructions(autoencoder, test_loader, num_images=10, device=device, epoch=epoch+1)
                visualizer.pca_2d_latent(autoencoder, test_loader, device=device, epoch=epoch+1)
                visualizer.umap_2d_latent(autoencoder, test_loader, device=device, epoch=epoch+1)
                visualizer.interpolate_2_to_5(autoencoder, test_loader, device=device, epoch=epoch+1)
                visualizer.visu_from_noise(autoencoder, device=device, latent_dim=latent_dim, epoch=epoch+1, num_images=10)
            visualizer.plot_losses(losses)
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {losses[-1]:.4f}")
    return losses