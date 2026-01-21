import torch
import argparse
from pathlib import Path

from src.training import train_autoencoder, train_vae, train_cvae
from src.ae import build_autoencoder
from src.vae import build_vae
from src.cvae import build_cvae
from src.data import get_fashion_mnist_dataloaders, get_mnist_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoencoder or Variational Autoencoder")
    parser.add_argument('--model', type=str, choices=['AE', 'VAE', 'CVAE'], default='AE', help="Model type to train: 'AE', 'VAE' or 'CVAE'")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist', help="Dataset to use: 'mnist' or 'fashion_mnist'")
    parser.add_argument('--latent_dim', type=int, default=32, help="Latent dimension size")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--visualise', action='store_true', help="Whether to generate visualizations during training")
    return parser.parse_args()


def main():

    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    if args.model == 'AE':
        if args.dataset == 'mnist':
            train_loader, test_loader = get_mnist_dataloaders(batch_size=128)

        elif args.dataset == 'fashion_mnist':
            train_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128)
        
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        model = build_autoencoder(latent_dim=args.latent_dim)
        print("Starting training...")
        train_autoencoder(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=1e-3,
                   latent_dim=args.latent_dim, device=device, visu_dir=f"{args.dataset}_autoencoder", visualise=args.visualise)
        
        save_path = Path('model/AE')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), save_path / f'encoder_{args.dataset}_{args.latent_dim}.pth')
        torch.save(model.decoder.state_dict(), save_path / f'decoder_{args.dataset}_{args.latent_dim}.pth')
        print(f"Model saved as '{save_path / f'encoder_{args.dataset}_{args.latent_dim}.pth'}' and '{save_path / f'decoder_{args.dataset}_{args.latent_dim}.pth'}'.")

    elif args.model == 'VAE':
        if args.dataset == 'mnist':
            train_loader, test_loader = get_mnist_dataloaders(batch_size=128, normalize=True)

        elif args.dataset == 'fashion_mnist':
            train_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128, normalize=True)
        
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        model = build_vae(latent_dim=args.latent_dim, mode="pp")
        print("Starting training...")
        train_vae(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=1e-3, latent_dim=args.latent_dim,
                   device=device, visu_dir=f"{args.dataset}_vae", visualise=args.visualise)
        
        save_path = Path('model/VAE')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), save_path / f'encoder_{args.dataset}.pth')
        torch.save(model.decoder.state_dict(), save_path / f'decoder_{args.dataset}.pth')
        print(f"Model saved as '{save_path / f'encoder_{args.dataset}.pth'}' and '{save_path / f'decoder_{args.dataset}.pth'}'.")

    elif args.model == 'CVAE':
        if args.dataset == 'mnist':
            train_loader, test_loader = get_mnist_dataloaders(batch_size=128, normalize=True)

        elif args.dataset == 'fashion_mnist':
            train_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128, normalize=True)
        
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        model = build_cvae(latent_dim=args.latent_dim)
        print("Starting training...")
        train_cvae(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=1e-3, latent_dim=args.latent_dim,
                    device=device, visu_dir=f"{args.dataset}_cvae", visualise=args.visualise)
        
        save_path = Path('model/CVAE')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), save_path / f'encoder_{args.dataset}.pth')
        torch.save(model.decoder.state_dict(), save_path / f'decoder_{args.dataset}.pth')
        print(f"Model saved as '{save_path / f'encoder_{args.dataset}.pth'}' and '{save_path / f'decoder_{args.dataset}.pth'}'.")

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

if __name__ == "__main__":
    # By default the script runs AE training on MNIST dataset with latent dim 32 for 50 epochs
    # if you want to run other configurations, use command line arguments
    # Example: python main.py --model VAE --dataset fashion_mnist --latent_dim 128 --epochs 50 
    # (train VAE on Fashion-MNIST with latent dim 128 for 50 epochs)
    
    main()