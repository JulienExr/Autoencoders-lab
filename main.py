import torch

from training import train_autoencoder, train_vae
from ae import build_autoencoder
from vae import build_vae
from data import get_mnist_dataloaders



def main_AE():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)

    autoencoder = build_autoencoder(latent_dim=256)

    print("Starting training...")
    train_autoencoder(autoencoder, train_loader, test_loader, num_epochs=20, learning_rate=1e-3, device=device)

    torch.save(autoencoder.encoder.state_dict(), 'model/AE/encoder.pth')
    torch.save(autoencoder.decoder.state_dict(), 'model/AE/decoder.pth')
    print("Model checkpoints saved as 'encoder.pth' and 'decoder.pth'.")

def main_VAE():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)

    vae = build_vae(latent_dim=32, mode ="pp")

    print("Start training")
    train_vae(vae, train_loader, test_loader, num_epochs=50, learning_rate=1e-3, latent_dim=32, device=device)
    torch.save(vae.encoder.state_dict(), 'model/VAE/encoder.pth')
    torch.save(vae.decoder.state_dict(), 'model/VAE/decoder.pth')
    print("Model checkpoints saved as 'encoder.pth' and 'decoder.pth'.")

if __name__ == "__main__":
    # By default the script runs AE training first, then VAE training.
    # If you want to run only one, comment the other line.
    main_AE()
    main_VAE()