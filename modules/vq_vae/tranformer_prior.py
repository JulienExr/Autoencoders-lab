import torch
import torch.nn.functional as F



class TransformerPrior(torch.nn.Module):
    def __init__(self, vocab_size=64, embedding_dim=64, num_head=4, num_layers=2, seq_len=49):
        super(TransformerPrior, self).__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, embedding_dim)

        self.position_embedding = torch.nn.Embedding(seq_len, embedding_dim)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_head, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = torch.nn.Linear(embedding_dim, vocab_size)

        self.register_buffer("mask", torch.triu(torch.ones(seq_len, seq_len)*float('-inf'), diagonal=1))

    def forward(self, x):
        batch_size, seq_len = x.size()
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(x) + self.position_embedding(positions)

        x = self.transformer(x, mask=self.mask[:seq_len, :seq_len])

        logits = self.output_layer(x)

        return logits

def build_transformer_prior(vocab_size=64, embedding_dim=64, num_head=4, num_layers=2, seq_len=49):
    model = TransformerPrior(vocab_size=vocab_size, embedding_dim=embedding_dim, num_head=num_head, num_layers=num_layers, seq_len=seq_len)
    return model