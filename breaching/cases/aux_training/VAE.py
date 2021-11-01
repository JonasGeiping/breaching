"""Train autoencoders."""

import torch
import torch.nn.functional as F

class VAE(torch.nn.Module):
    """Closely following https://github.com/pytorch/examples/blob/master/vae/main.py."""
    def __init__(self, feature_model, decoder, kl_coef=1.0):
        super(VAE, self).__init__()

        self.encoder = feature_model
        self.decoder = decoder

        self.kl_coef = kl_coef

    def reparameterize(self, mu, logvar, noise_level=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * noise_level
        return mu + eps * std

    def forward(self, x, noise_level=1.0):
        code = self.encoder(x).flatten(start_dim=1)
        cutoff = code.shape[1] // 2
        mu, logvar = code[:, :cutoff], code[:, cutoff:]
        z = self.reparameterize(mu, logvar, noise_level)
        return self.decoder(torch.cat([z] * 2, dim=1)), mu, logvar

    def loss(self, x, recon_x, mu, logvar, data_mean=0.0, data_std=1.0):
        """Based on https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py#L93.
        Compare BCE on unnormalized images."""
        B = x.shape[0]
        bce = F.binary_cross_entropy(recon_x * data_std + data_mean, x * data_std + data_mean, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # print(bce, kl)
        return bce + self.kl_coef * kl

def train_encoder_decoder(encoder, decoder, dataloader, setup, arch='VAE'):
    """Train a VAE."""
    epochs = 50
    lr = 1e-3
    data_mean = torch.as_tensor(dataloader.dataset.mean, **setup)[None, :, None, None]
    data_std = torch.as_tensor(dataloader.dataset.std, **setup)[None, :, None, None]

    if arch == 'VAE':
        model = VAE(encoder, decoder, kl_coef=1.0)
        model.to(**setup)
    else:
        raise ValueError('Invalid model.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for epoch in range(epochs):
        epoch_loss, epoch_mse = 0, 0
        for idx, (data, label) in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            data = data.to(**setup)

            reconstructed_data, mu, logvar = model(data, noise_level=1.0)
            loss = model.loss(data, reconstructed_data, mu, logvar, data_mean=data_mean, data_std=data_std)
            loss.backward()
            with torch.no_grad():
                epoch_loss += loss.detach()
                epoch_mse += F.mse_loss(data, reconstructed_data)
                optimizer.step()
            print(f'Epoch {epoch}_{idx}: Avg. Loss: {epoch_loss / (idx + 1)}. Avg. MSE: {epoch_mse / (idx + 1)}')


def status_message(optimizer, stats, step):
    """A basic console printout."""
    current_lr = f'{optimizer.param_groups[0]["lr"]:.4f}'

    def _maybe_print(key):
        return stats[key][-1] if len(stats[key]) > 0 else float('NaN')

    msg = f'Step: {step:<4}| lr: {current_lr} | Time: {stats["train_time"][-1]:4.2f}s |'
    msg += f'TRAIN loss {stats["train_loss"][-1]:7.4f} | TRAIN Acc: {stats["train_acc"][-1]:7.2%} |'
    msg += f'VAL loss {_maybe_print("valid_loss"):7.4f} | VAL Acc: {_maybe_print("valid_acc"):7.2%} |'
    return msg
