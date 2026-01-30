import torch

def gn(c):
    for g in (32, 16, 8, 4, 2, 1):
        if c % g == 0:
            return torch.nn.GroupNorm(g, c)
    return torch.nn.GroupNorm(1, c)

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)



class ResBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.gn1 = gn(channels)
        self.act1 = torch.nn.SiLU()
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.gn2 = gn(channels)
        self.act2 = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
    
    def forward(self, x):
        out = self.gn1(x)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.gn2(out)
        out = self.act2(out)
        out = self.conv2(out)
        return x + out

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar
    

class VAE_Encoder(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # 1x28x28 -> 32x14x14
        self.gn1 = gn(32)
        self.relu1 = torch.nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32x14x14 -> 64x7x7
        self.gn2 = gn(64)
        self.relu2 = torch.nn.LeakyReLU(0.2, inplace= True)
        self.flatten = torch.nn.Flatten()
        self.fc_mu = torch.nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = torch.nn.Linear(64 * 7 * 7, latent_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.relu2(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
class VAE_Decoder(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE_Decoder, self).__init__()
        self.linear = torch.nn.Linear(latent_dim, 64 * 7 * 7) 
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')                 # 64x7x7 -> 64x14x14
        self.deconv1 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # 64x14x14 -> 32x14x14
        self.gn1 = gn(32)
        self.relu1 = torch.nn.LeakyReLU(0.2, inplace=True)
        self.deconv2 = torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)   # 32x28x28 -> 1x28x28
        self.output_activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.up(x)
        x = self.deconv1(x)
        x = self.gn1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.deconv2(x)
        x = self.output_activation(x)
        return x

class VAE_EncoderPP(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE_EncoderPP, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1) # 1x28x28 -> 32x14x14
        self.res1 = ResBlock(32)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # 32x14x14 -> 64x7x7
        self.res2 = ResBlock(64)

        self.res3 = ResBlock(64)

        self.flatten = torch.nn.Flatten()
        self.fc_mu = torch.nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = torch.nn.Linear(64 * 7 * 7, latent_dim)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class VAE_DecoderPP(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE_DecoderPP, self).__init__()
        self.linear = torch.nn.Linear(latent_dim, 64 * 7 * 7) 
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')                 # 64x7x7 -> 64x14x14

        self.res1 = ResBlock(64)
        self.deconv1 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)  # 64x14x14 -> 32x14x14

        self.res2 = ResBlock(32)
        self.deconv2 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)   # 32x28x28 -> 16x28x28

        self.res3 = ResBlock(16)
        self.deconv3 = torch.nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)   # 16x28x28 -> 1x28x28

        self.output_activation = torch.nn.Tanh()
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 64, 7, 7)
        
        x = self.res1(x)
        x = self.up(x)
        x = self.deconv1(x)

        x = self.res2(x)
        x = self.up(x)
        x = self.deconv2(x)

        x = self.res3(x)
        x = self.deconv3(x)
        x = self.output_activation(x)

        return x

def build_vae(latent_dim, mode = "default"):
    if mode == "pp":
        encoder = VAE_EncoderPP(latent_dim)
        decoder = VAE_DecoderPP(latent_dim)
    else:
        encoder = VAE_Encoder(latent_dim)
        decoder = VAE_Decoder(latent_dim)
    
    vae = VAE(encoder, decoder)
    return vae




class VAE_CIFAR_Encoder(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE_CIFAR_Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1) # 3x32x32 -> 64x16x16
        self.res1 = ResBlock(64)
        self.res1_b = ResBlock(64)

        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 64x16x16 -> 128x8x8
        self.res2 = ResBlock(128)
        self.res2_b = ResBlock(128)
        
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # 128x8x8 -> 256x4x4
        self.res3 = ResBlock(256)
        self.res3_b = ResBlock(256)

        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1) # 256x4x4 -> 512x4x4
        self.res4 = ResBlock(512)
        self.res4_b = ResBlock(512)

        self.flatten = torch.nn.Flatten()
        self.fc_mu = torch.nn.Linear(512 * 4 * 4, latent_dim,)
        self.fc_mu.weight.data.normal_(0, 0.01)
        self.fc_logvar = torch.nn.Linear(512 * 4 * 4, latent_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res1_b(x)

        x = self.conv2(x)
        x = self.res2(x)
        x = self.res2_b(x)
        
        x = self.conv3(x)
        x = self.res3(x)
        x = self.res3_b(x)

        x = self.conv4(x)
        x = self.res4(x)
        x = self.res4_b(x)
        
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10.0, max=2)
        
        return mu, logvar

class VAE_CIFAR_Decoder(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE_CIFAR_Decoder, self).__init__()
        self.linear = torch.nn.Linear(latent_dim, 512 * 4 * 4) 

        self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)                 # 512x4x4 -> 256x8x8
        self.res1 = ResBlock(512)
        self.res1_b = ResBlock(512)
        self.deconv1 = torch.nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect')  # 512x8x8 -> 256x8x8
        self.activation1 = torch.nn.SiLU()

        self.up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)                 # 256x8x8 -> 256x16x16
        self.res2 = ResBlock(256)
        self.res2_b = ResBlock(256)
        self.deconv2 = torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect')   # 256x16x16 -> 128x16x16
        self.activation2 = torch.nn.SiLU()

        self.up3 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)                 # 128x16x16 -> 128x32x32
        self.res3 = ResBlock(128)
        self.res3_b = ResBlock(128)
        self.deconv3 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect')   # 128x32x32 -> 64x32x32
        self.activation3 = torch.nn.SiLU()

        self.res4 = ResBlock(64)
        self.res4_b = ResBlock(64)
        self.deconv4 = torch.nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')   # 64x32x32 -> 3x32x32

        self.output_activation = torch.nn.Tanh()
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.size(0), 512, 4, 4)

        x = self.res1(x)
        x = self.res1_b(x)
        x = self.up(x)
        x = self.deconv1(x)
        x = self.activation1(x)

        x = self.res2(x)
        x = self.res2_b(x)
        x = self.up2(x)
        x = self.deconv2(x)
        x = self.activation2(x)

        x = self.res3(x)
        x = self.res3_b(x)
        x = self.up3(x)
        x = self.deconv3(x)
        x = self.activation3(x)

        x = self.res4(x)
        x = self.res4_b(x)
        x = self.deconv4(x)

        x = self.output_activation(x)
        return x
    
def build_vae_cifar(latent_dim=256):
    encoder = VAE_CIFAR_Encoder(latent_dim)
    decoder = VAE_CIFAR_Decoder(latent_dim)
    vae = VAE(encoder, decoder)
    vae.apply(init_weights)
    return vae