import torch
import torch.nn.functional as F

def gn(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return torch.nn.GroupNorm(g, c)
    return torch.nn.GroupNorm(1, c)


class ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.gn1 = gn(channels)
        self.act1 = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn2 = gn(channels)
        self.act2 = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.gn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.gn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        return x + out

class VectorQuantizer(torch.nn.Module):
    def __init__(self, num_embeddings=64, embedding_dim=32, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1).contiguous() # B x C x H x W -> B x H x W x C
        input_shape = x.shape

        flat_x = x.view(-1, self.embedding_dim)

        distances = torch.sum(flat_x**2, dim=1, keepdim=True) + \
                    torch.sum(self.embedding.weight**2, dim=1) - \
                    2 * torch.matmul(flat_x, self.embedding.weight.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        quantized = self.embedding(encoding_indices).view(input_shape)
        
        q_latent_loss = F.mse_loss(quantized.detach(), x)
        e_latent_loss = F.mse_loss(quantized, x.detach())

        loss = e_latent_loss + self.commitment_cost * q_latent_loss

        quantized = x + (quantized - x).detach()

        quantized = quantized.permute(0, 3, 1, 2).contiguous() # B x H x W x C -> B x C x H x W

        return quantized, loss, encoding_indices

class VQVAE(torch.nn.Module):
    def __init__(self, encoder, decoder, vector_quantizer):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vector_quantizer = vector_quantizer
    
    def forward(self, x):
        encoded = self.encoder(x)
        quantized, loss, encoding_indices = self.vector_quantizer(encoded)
        x_recon = self.decoder(quantized)
        return x_recon, loss

    def get_codebook_indices(self, x):
        with torch.no_grad():
            encoded = self.encoder(x)
            _, _, encoding_indices = self.vector_quantizer(encoded)
        return encoding_indices.view(x.size(0), -1)

class VQVAEEncoder(torch.nn.Module):
    def __init__(self, embedding_dim=32):
        super(VQVAEEncoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # 1x28x28 -> 32x14x14
        self.res1 = ResBlock(32)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32x14x14 -> 64x7x7
        self.res2 = ResBlock(64)

        self.conv3 = torch.nn.Conv2d(64, embedding_dim, kernel_size=3, stride=1, padding=1) # 64x7x7 -> embedding_dimx7x7
        self.res3 = ResBlock(embedding_dim)   # embedding_dimx7x7


    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)

        x = self.conv2(x)
        x = self.res2(x)

        x = self.conv3(x)
        x = self.res3(x)

        return x


class VQVAEDecoder(torch.nn.Module):
    def __init__(self, embedding_dim=32):
        super(VQVAEDecoder, self).__init__()
        
        self.res1 = ResBlock(embedding_dim)
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')                 # 32 x7x7 -> 32x14x14
        self.deconv1 = torch.nn.Conv2d(embedding_dim, 64, kernel_size=3, stride=1, padding=1)  # 32x14x14 -> 64x14x14
        self.res2 = ResBlock(64)
        self.up2 = torch.nn.Upsample(scale_factor=2, mode='nearest')                 # 64x14x14 -> 64x28x28
        self.deconv2 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)   # 64x28x28 -> 32x28x28
        self.res3 = ResBlock(32)
        self.output_conv = torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)   # 32x28x28 -> 1x28x28
        self.output_activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.res1(x)
        x = self.up(x)
        x = self.deconv1(x)

        x = self.res2(x)
        x = self.up2(x)
        x = self.deconv2(x)

        x = self.res3(x)

        x = self.output_conv(x)
        x = self.output_activation(x)
        return x
    
def build_vqvae(embedding_dim=32, num_embeddings=64, commitment_cost=0.25):
    encoder = VQVAEEncoder(embedding_dim=embedding_dim)
    decoder = VQVAEDecoder(embedding_dim=embedding_dim)
    vector_quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
    vqvae = VQVAE(encoder, decoder, vector_quantizer)
    return vqvae