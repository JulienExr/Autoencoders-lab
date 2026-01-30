import torch
import argparse
from pathlib import Path

from modules.autoencoder.training import train_autoencoder
from modules.autoencoder.ae import build_autoencoder, build_autoencoder_cifar
from modules.vae.training import train_vae
from modules.vae.vae import build_vae, build_vae_cifar
from modules.cvae.training import train_cvae
from modules.cvae.cvae import build_cvae
from modules.vq_vae.training import train_transformer_prior, train_vqvae
from modules.vq_vae.vq_vae import build_vqvae
from modules.vq_vae.tranformer_prior import build_transformer_prior
from src.data import get_cifar10_dataloaders, get_fashion_mnist_dataloaders, get_mnist_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoencoder or Variational Autoencoder")
    parser.add_argument('--model', type=str, choices=['AE', 'VAE', 'CVAE', 'VQ-VAE'], default='AE', help="Model type to train: 'AE', 'VAE', 'CVAE' or 'VQ-VAE'")
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist', 'cifar10'], default='mnist', help="Dataset to use: 'mnist', 'fashion_mnist' or 'cifar10'")
    parser.add_argument('--latent_dim', type=int, default=32, help="Latent dimension size")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--visualize', action='store_true', help="Whether to generate visualizations during training")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate for training")
    return parser.parse_args()


def main():

    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")


    if args.model == 'AE':
        if args.dataset == 'mnist':
            train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
            model = build_autoencoder(latent_dim=args.latent_dim)
            print("Starting training...")
            train_autoencoder(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=args.lr,
                latent_dim=args.latent_dim, device=device, visu_dir=f"{args.dataset}_autoencoder", visualize=args.visualize)

        elif args.dataset == 'fashion_mnist':
            train_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128)
            model = build_autoencoder(latent_dim=args.latent_dim)
            print("Starting training...")
            train_autoencoder(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=args.lr,
                   latent_dim=args.latent_dim, device=device, visu_dir=f"{args.dataset}_autoencoder", visualize=args.visualize)

        elif args.dataset == 'cifar10':
            train_loader, test_loader = get_cifar10_dataloaders(batch_size=128, normalize=True)
            model = build_autoencoder_cifar(latent_dim=args.latent_dim)
            print("Starting training...")
            train_autoencoder(model, train_loader, test_loader, num_epochs=args.epochs,
                            learning_rate=args.lr,latent_dim=args.latent_dim, device=device, visu_dir=f"{args.dataset}_autoencoder",
                            visualize=args.visualize)

        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
                
        save_path = Path('models/AE')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), save_path / f'encoder_{args.dataset}_{args.latent_dim}.pth')
        torch.save(model.decoder.state_dict(), save_path / f'decoder_{args.dataset}_{args.latent_dim}.pth')
        print(f"Model saved as '{save_path / f'encoder_{args.dataset}_{args.latent_dim}.pth'}' and '{save_path / f'decoder_{args.dataset}_{args.latent_dim}.pth'}'.")

    elif args.model == 'VAE':
        if args.dataset == 'mnist':
            train_loader, test_loader = get_mnist_dataloaders(batch_size=128, normalize=True)
            model = build_vae(latent_dim=args.latent_dim, mode="pp")
            print("Starting training...")
            train_vae(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=args.lr, latent_dim=args.latent_dim,
                   device=device, visu_dir=f"{args.dataset}_vae", visualize=args.visualize)
            
        elif args.dataset == 'fashion_mnist':
            train_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128, normalize=True)
            model = build_vae(latent_dim=args.latent_dim, mode="pp")
            print("Starting training...")
            train_vae(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=args.lr, latent_dim=args.latent_dim,
                   device=device, visu_dir=f"{args.dataset}_vae", visualize=args.visualize)  
            
        elif args.dataset == 'cifar10':
            train_loader, test_loader = get_cifar10_dataloaders(batch_size=128, normalize=True)
            model = build_vae_cifar(latent_dim=args.latent_dim)
            print("Starting training on cifar10...")
            train_vae(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=args.lr, latent_dim=args.latent_dim,
                 device=device, visu_dir=f"{args.dataset}_vae", visualize=args.visualize)
            
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        

        
        save_path = Path('models/VAE')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), save_path / f'encoder_{args.dataset}_{args.latent_dim}.pth')
        torch.save(model.decoder.state_dict(), save_path / f'decoder_{args.dataset}_{args.latent_dim}.pth')
        print(f"Model saved as '{save_path / f'encoder_{args.dataset}_{args.latent_dim}.pth'}' and '{save_path / f'decoder_{args.dataset}_{args.latent_dim}.pth'}'.")

    elif args.model == 'CVAE':
        if args.dataset == 'mnist':
            train_loader, test_loader = get_mnist_dataloaders(batch_size=128, normalize=True)

        elif args.dataset == 'fashion_mnist':
            train_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128, normalize=True)
        
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        
        model = build_cvae(latent_dim=args.latent_dim)
        print("Starting training...")
        train_cvae(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=args.lr      , latent_dim=args.latent_dim,
                    device=device, visu_dir=f"{args.dataset}_cvae", visualize=args.visualize)
        
        save_path = Path('models/CVAE')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), save_path / f'encoder_{args.dataset}.pth')
        torch.save(model.decoder.state_dict(), save_path / f'decoder_{args.dataset}.pth')
        print(f"Model saved as '{save_path / f'encoder_{args.dataset}.pth'}' and '{save_path / f'decoder_{args.dataset}.pth'}'.")

    elif args.model == 'VQ-VAE':
        if args.dataset == 'mnist':
            train_loader, test_loader = get_mnist_dataloaders(batch_size=128, normalize=True)
        elif args.dataset == 'fashion_mnist':
            train_loader, test_loader = get_fashion_mnist_dataloaders(batch_size=128, normalize=True)
        num_embeddings = 64
        embedding_dim = 64
        commitment_cost = 0.25
        
        model = build_vqvae(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
        print("Starting training...")
        train_vqvae(model, train_loader, test_loader, num_epochs=args.epochs, learning_rate=args.lr,
                     visu_dir=f"{args.dataset}_vqvae", visualize=args.visualize)
        save_path = Path('models/VQ-VAE')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), save_path / f'encoder_{args.dataset}.pth')
        torch.save(model.decoder.state_dict(), save_path / f'decoder_{args.dataset}.pth')
        torch.save(model.vector_quantizer.state_dict(), save_path / f'vq_{args.dataset}.pth')
        print(f"Model saved as '{save_path / f'encoder_{args.dataset}.pth'}' and '{save_path / f'decoder_{args.dataset}.pth'}'.")

        transformer = build_transformer_prior(vocab_size=num_embeddings, embedding_dim=embedding_dim, num_head=4, num_layers=6, seq_len=49)
        train_transformer_prior(transformer, model, train_loader, num_epochs=args.epochs, learning_rate=args.lr, device=device)

        save_path = Path('models/TransformerPrior')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(transformer.state_dict(), save_path / f'transformer_{args.dataset}.pth')
        print(f"Transformer prior model saved as '{save_path / f'transformer_{args.dataset}.pth'}'.")

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

if __name__ == "__main__":
    # By default the script runs AE training on MNIST dataset with latent dim 32 for 50 epochs
    # if you want to run other configurations, use command line arguments
    # Example: python main.py --model VAE --dataset fashion_mnist --latent_dim 128 --epochs 50 
    # (train VAE on Fashion-MNIST with latent dim 128 for 50 epochs)
    
    main()