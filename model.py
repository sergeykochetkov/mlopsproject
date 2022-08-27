'''
model for prediction stock returns
'''
import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    '''
    encoding time of lag feature
    '''

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, 1, d_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, features_x: Tensor) -> Tensor:
        """
        Args:
            features_x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        features_x = features_x + self.pos_encoding[:features_x.size(0)]
        return self.dropout(features_x)


class TransformerModel(nn.Module):
    '''
    transformer for timeseries prediction
    '''

    def __init__(self, length: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self._params = {'length': length, 'd_model': d_model, 'nhead': nhead, 'd_hid': d_hid,
                        'nlayers': nlayers, 'dropout': dropout}

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Sequential(
            nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model),
            nn.ReLU(), nn.Linear(d_model, d_model))
        self.d_model = self._params['d_model']
        self.length = self._params['length']
        self.decoder = nn.Linear(d_model * length, 1)

        self.init_weights()

    def init_weights(self) -> None:
        '''
        custom weight initialization
        '''
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = src.transpose(0, 1)  # [seq_len, batch_size]
        src = torch.unsqueeze(src, dim=-1)
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.transpose(0, 1)  # [batch_size, seq_len]
        output = output.flatten(-2, -1)
        output = self.decoder(output)
        return torch.squeeze(torch.sigmoid(output) * 2 - 1, dim=-1)
