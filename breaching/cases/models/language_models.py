"""Code entirely taken from the pytorch example on language modeling at
https://github.com/pytorch/examples/blob/master/word_language_model/model.py
"""

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LinearModel(torch.nn.Module):
    """Container with just an encoder and a decoder."""

    def __init__(self, vocab_size, embedding_size, tie_weights=True):
        super().__init__()
        self.encoder = torch.nn.Embedding(vocab_size, embedding_size)
        self.decoder = torch.nn.Linear(embedding_size, vocab_size)
        if tie_weights:
            self.decoder.weight = self.encoder.weight

    def forward(self, input_ids, inputs_embeds=None, *args, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.encoder(input_ids)
        return self.decoder(inputs_embeds)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntokens, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.ntoken = ntokens
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntokens, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                )
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True)
        self.decoder = nn.Linear(nhid, ntokens)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError("When using the tied flag, nhid must be equal to emsize")
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input_ids, hiddens, **kwargs):
        emb = self.drop(self.encoder(input_ids))
        output, hidden = self.rnn(emb, hiddens)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        # return F.log_softmax(decoded, dim=1), hidden
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == "LSTM":
            return (weight.new_zeros(bsz, self.nlayers, self.nhid), weight.new_zeros(bsz, self.nlayers, self.nhid))
        else:
            return weight.new_zeros(bsz, self.nlayers, self.nhid)


# Temporarily leave PositionalEmbedding module here. Will be moved somewhere else.
class PositionalEmbedding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEmbedding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class LearnablePositionalEmbedding(torch.nn.Module):
    """Shorthand for a learnable embedding."""

    def __init__(self, embed_dim, max_position_embeddings=1024, dropout=0.0):
        super().__init__()
        self.embedding = torch.nn.Embedding(max_position_embeddings, embed_dim)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, input_embeddings):
        """This is a batch-first implementation"""
        position_ids = torch.arange(input_embeddings.shape[1], device=self.embedding.weight.device)
        position_embeddings = self.embedding(position_ids[None, :])
        return self.dropout(input_embeddings + position_embeddings)


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(
        self, ntokens, ninp, nhead, nhid, nlayers, dropout=0.5, positional_embedding="fixed", tie_weights=False
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.src_mask = None
        if positional_embedding == "fixed":
            self.pos_encoder = PositionalEmbedding(ninp, dropout)
        else:
            self.pos_encoder = LearnablePositionalEmbedding(ninp, dropout=dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntokens, ninp)
        self.encoder.weight.data *= math.sqrt(ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntokens)
        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input_ids, has_mask=False, inputs_embeds=None, **kwargs):
        """Can utilize input embeddings directly instead of inputs."""
        if has_mask:
            device = input_ids.device
            if self.src_mask is None or self.src_mask.shape[1] != input_ids.shape[1]:
                mask = self._generate_square_subsequent_mask(input_ids.shape[1]).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        if inputs_embeds is None:
            inputs = self.encoder(input_ids)
        else:
            inputs = inputs_embeds
        inputs = self.pos_encoder(inputs)
        output = self.transformer_encoder(inputs, self.src_mask)
        output = self.decoder(output)
        return output
