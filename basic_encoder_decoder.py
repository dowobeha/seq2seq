from typing import List, Tuple

import torch
from torch import nn
from torch.nn.functional import log_softmax, relu, softmax

from utils import verify_shape

# Note by Lane (2020-05-23):
#
# I don't remember whether this code is functional or not.
# At a minimum, it needs to be inspected and tested.
# If it's not currently functional, it probably shouldn't take too much work to get it functional


class EncoderRNN(nn.Module):
    def __init__(self, *, input_size: int, embedding_size: int,
                 hidden_size: int):
        super(EncoderRNN, self).__init__()
        self.num_hidden_layers: int = 1
        self.hidden_size: int = hidden_size
        self.embedding: nn.Embedding = nn.Embedding(input_size, embedding_size)
        self.rnn: nn.RNN = nn.RNN(embedding_size, hidden_size, num_layers=self.num_hidden_layers)

    def forward(self, input_tensor: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        embedded: torch.Tensor = self.embedding(input_tensor).unsqueeze(dim=0)  # <--- Replacement to enable batching

        output, hidden = self.rnn(embedded, hidden)

        return output, hidden

    def init_hidden(self, *, batch_size: int = 1, device: torch.device) -> torch.Tensor:

        hidden: torch.Tensor = torch.zeros(self.rnn.num_layers, batch_size, self.hidden_size, device=device)

        return hidden


class Decoder(nn.Module):

    def __init__(self, *,
                 embedding_size: int,
                 hidden_size: int,
                 output_size: int):
        super().__init__()

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self,
                input_tensor: torch.Tensor,
                hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                batch_size: int):

        embedded = self.embedding(input_tensor)

        output, hidden = self.rnn(embedded, hidden)

        verify_shape(tensor=output, expected=[1, batch_size, self.decoder_hidden_size])
        verify_shape(tensor=hidden, expected=[self.gru.num_layers, batch_size, self.decoder_hidden_size])

        output = output.squeeze(dim=0)

        verify_shape(tensor=output, expected=[batch_size, self.decoder_hidden_size])

        output = log_softmax(self.out(output), dim=1)

        verify_shape(tensor=attn_weights, expected=[batch_size, self.max_src_length, 1])
        attn_weights = attn_weights.squeeze(dim=2)

        verify_shape(tensor=output, expected=[batch_size, self.output_size])
        verify_shape(tensor=hidden, expected=[self.gru.num_layers, batch_size, self.decoder_hidden_size])
        verify_shape(tensor=attn_weights, expected=[batch_size, self.max_src_length])

        # print(f"output.shape={output.shape}\t\thidden.shape={hidden.shape}\t\toutput[0].shape={output[0].shape}")

        # output.shape:             [seq_length=1, batch_size=1, decoder_hidden_size]
        # hidden.shape:             [num_layers=1, batch_size=1, decoder_hidden_size]
        #
        # output[0].shape:                        [batch_size=1, decoder_hidden_size]
        # output = log_softmax(self.out(output[0]), dim=1)

        # print(f"output.shape={output.shape}\t\thidden.shape={hidden.shape}\t\tattn_weights.shape={attn_weights.shape}")

        # output.shape:                           [batch_size=1, decoder_output_size]
        # hidden.shape:             [num_layers=1, batch_size=1, decoder_hidden_size]
        # attn_weights:                           [batch_size=1, encoder_max_len]
        #
        return output, hidden, attn_weights

    def init_hidden(self, *, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.gru.num_layers, batch_size, self.decoder_hidden_size, device=device)




def encoder_decoder(encoder: EncoderRNN, input_sequence: torch.LongTensor) -> torch.Tensor:
    sequence_length: int = input_sequence.shape[0]
    batch_size: int = input_sequence.shape[1]
    device: torch.device = input_sequence.device

    encoder_hidden = encoder.init_hidden(batch_size=batch_size,
                                         device=device)

    encoder_outputs = torch.zeros(sequence_length,
                                  batch_size,
                                  encoder.hidden_size,
                                  device=device)  # shape: [max_src_len, hidden_size]

    verify_shape(tensor=input_sequence, expected=[sequence_length, batch_size])
    verify_shape(tensor=encoder_hidden, expected=[encoder.num_hidden_layers, batch_size, encoder.hidden_size])
    verify_shape(tensor=encoder_outputs, expected=[sequence_length, batch_size, encoder.hidden_size])

    for src_index in range(sequence_length):

        input_token_tensor: torch.Tensor = input_sequence[src_index]

        verify_shape(tensor=input_token_tensor, expected=[batch_size])
        verify_shape(tensor=encoder_hidden, expected=[encoder.num_hidden_layers, batch_size, encoder.hidden_size])

        encoder_output, encoder_hidden = encoder(input_token_tensor, encoder_hidden)

        verify_shape(tensor=encoder_hidden, expected=[encoder.num_hidden_layers, batch_size, encoder.hidden_size])
        verify_shape(tensor=encoder_output, expected=[1, batch_size, encoder.hidden_size])

        verify_shape(tensor=encoder_output[0], expected=[batch_size, encoder.hidden_size])
        verify_shape(tensor=encoder_outputs[src_index], expected=[batch_size, encoder.hidden_size])

        encoder_outputs[src_index] = encoder_output[0]

    verify_shape(tensor=encoder_outputs, expected=[sequence_length, batch_size, encoder.hidden_size])
    return encoder_outputs
