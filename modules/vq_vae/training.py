import torch
import torch.nn.functional as F

from src.visualization import Visualizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_vqvae(vqvae, dataloader, test_loader, num_epochs=10, learning_rate=1e-3, visu_dir="mnist_vqvae", visualize = True):
    vqvae.to(device)
    vqvae.train()

    optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    recon_losses = []
    vq_losses = []
    losses = []
    visualizer = Visualizer(directory=visu_dir, model=vqvae)
    latent_shape = None

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for (idx, data) in enumerate(dataloader):
            images, labels = data
            images = images.to(device)

            decoded, vq_loss = vqvae(images)
            recon_loss = criterion(decoded, images)

            loss = recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if (idx + 1) % 100 == 0 or idx == len(dataloader)-1:
                print(f"  Batch [{idx+1}/{len(dataloader)}], recon_Loss: {recon_loss.item():.4f} | VQ_Loss: {vq_loss.item():.4f}", end='\r', flush=True)
        
        losses.append(epoch_loss / len(dataloader))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("\nGenerating visualizations...", visualize)
            if visualize:
                visualizer.visualize_reconstructions(vqvae, test_loader, num_images=10, device=device, epoch=epoch+1)
                if latent_shape is None:
                    try:
                        sample_images, _ = next(iter(test_loader))
                        sample_images = sample_images.to(device)[:1]
                        with torch.no_grad():
                            encoded = vqvae.encoder(sample_images)
                        latent_shape = tuple(encoded.shape[1:])
                    except Exception:
                        latent_shape = None
                visualizer.visu_from_noise(vqvae, device=device, latent_dim=latent_shape, epoch=epoch+1, num_images=10)
            visualizer.plot_losses(losses)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
    
    return losses



def train_transformer_prior(transformer, vqvae, dataloader, num_epochs=10, learning_rate=3e-4, device=device, visualize=True, visu_dir="transformer_prior"):
    transformer.to(device)
    transformer.train()

    vqvae.to(device)
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate)
    
    num_embeddings = vqvae.vector_quantizer.num_embeddings
    vocab_size = transformer.token_embedding.num_embeddings
    use_sos = vocab_size == (num_embeddings + 1)
    sos_token = num_embeddings if use_sos else None
    max_seq_len = transformer.position_embedding.num_embeddings

    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    visualizer = Visualizer(directory=visu_dir, model=transformer)
    latent_shape = None
    
    print("Starting training of Transformer prior...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for (idx, data) in enumerate(dataloader):
            images, labels = data
            images = images.to(device)
            batch_size = images.size(0)

            with torch.no_grad():
                
                indices = vqvae.get_codebook_indices(images)
                indices = indices.view(batch_size, -1)
                if latent_shape is None:
                    encoded = vqvae.encoder(images[:1])
                    latent_shape = tuple(encoded.shape[1:])
                max_index = int(indices.max().item())
                if max_index >= vocab_size:
                    raise ValueError(
                        f"Transformer vocab_size ({vocab_size}) is smaller than codebook indices max ({max_index}). "
                        f"Rebuild TransformerPrior with vocab_size={num_embeddings + 1} (for SOS) or >= {num_embeddings}."
                    )

            if use_sos:
                sos_col = torch.full((batch_size, 1), sos_token, device=device, dtype=torch.long)
                valid_seq = torch.cat([sos_col, indices], dim=1)
                input_seq = valid_seq[:, :-1]  # [SOS, idx_1, idx_2, ..., idx_n-1]
                target_seq = valid_seq[:, 1:]  # [idx_1, idx_2, ..., idx_n]
            else:
                input_seq = indices[:, :-1]
                target_seq = indices[:, 1:]

            if input_seq.size(1) > max_seq_len:
                input_seq = input_seq[:, :max_seq_len]
                target_seq = target_seq[:, :max_seq_len]

            logits = transformer(input_seq)

            loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (idx + 1) % 100 == 0 or idx == len(dataloader)-1:
                print(f"  Batch [{idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}", end='\r', flush=True)
        losses.append(epoch_loss / len(dataloader))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print("\nGenerating transformer prior visualizations...", visualize)
            if visualize and latent_shape is not None:
                visualizer.visu_from_transformer_prior(
                    transformer,
                    vqvae,
                    device=device,
                    epoch=epoch + 1,
                    num_images=10,
                    latent_shape=latent_shape,
                    temperature=1.0,
                    top_k=128,
                )
            visualizer.plot_losses(losses)

        print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
    return losses

