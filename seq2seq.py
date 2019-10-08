from typing import List, Tuple

import torch
from torch import nn
from torch.nn.functional import log_softmax, relu, softmax

from utils import verify_shape


class EncoderRNN(nn.Module):
    def __init__(self, *, input_size: int, embedding_size: int, hidden_size: int, num_hidden_layers: int):
        super(EncoderRNN, self).__init__()
        self.num_hidden_layers: int = num_hidden_layers
        self.hidden_size: int = hidden_size
        self.embedding: nn.Embedding = nn.Embedding(input_size, embedding_size)
        self.gru: nn.GRU = nn.GRU(embedding_size, hidden_size, num_layers=num_hidden_layers)

    def forward(self,
                input_tensor: torch.Tensor,
                hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # ignore[override]
        #print(f"input_tensor.shape={input_tensor.shape}")
        embedded: torch.Tensor = self.embedding(input_tensor).unsqueeze(dim=0)  # <--- Replacement to enable batching
        #print(f"embedded.shape={embedded.shape}")
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def init_hidden(self, *, batch_size: int = 1, device: torch.device) -> torch.Tensor:
        # hidden.shape:           [num_layers=1, batch_size=1, hidden_size=256]
        hidden: torch.Tensor = torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, device=device)
        return hidden

    def encode_sequence(self, input_sequence: torch.LongTensor) -> torch.Tensor:
        sequence_length: int = input_sequence.shape[0]
        batch_size: int = input_sequence.shape[1]
        device: torch.device = input_sequence.device

        encoder_hidden = self.init_hidden(batch_size=batch_size,
                                          device=device)

        encoder_outputs = torch.zeros(sequence_length,
                                      batch_size,
                                      self.hidden_size,
                                      device=device)  # shape: [max_src_len, hidden_size]

        verify_shape(tensor=input_sequence, expected=[sequence_length, batch_size])
        verify_shape(tensor=encoder_hidden, expected=[self.num_hidden_layers, batch_size, self.hidden_size])
        verify_shape(tensor=encoder_outputs, expected=[sequence_length, batch_size, self.hidden_size])

        for src_index in range(sequence_length):

            input_token_tensor: torch.Tensor = input_sequence[src_index]

            verify_shape(tensor=input_token_tensor, expected=[batch_size])
            verify_shape(tensor=encoder_hidden, expected=[self.num_hidden_layers, batch_size, self.hidden_size])

            encoder_output, encoder_hidden = self(input_token_tensor, encoder_hidden)

            verify_shape(tensor=encoder_hidden, expected=[self.num_hidden_layers, batch_size, self.hidden_size])
            verify_shape(tensor=encoder_output, expected=[1, batch_size, self.hidden_size])

            verify_shape(tensor=encoder_output[0], expected=[batch_size, self.hidden_size])
            verify_shape(tensor=encoder_outputs[src_index], expected=[batch_size, self.hidden_size])

            encoder_outputs[src_index] = encoder_output[0]

        verify_shape(tensor=encoder_outputs, expected=[sequence_length, batch_size, self.hidden_size])
        return encoder_outputs


class AttnDecoderRNN(nn.Module):
    def __init__(self, *,
                 embedding_size: int,
                 decoder_hidden_size: int,
                 encoder_hidden_size: int,
                 max_src_length: int,
                 num_hidden_layers: int,
                 output_size: int,
                 dropout_p: float = 0.1):
        super(AttnDecoderRNN, self).__init__()
        self.decoder_hidden_size: int = decoder_hidden_size
        self.encoder_hidden_size: int = encoder_hidden_size
        self.embedding_size: int = embedding_size
        self.output_size: int = output_size
        self.dropout_p: float = dropout_p
        self.max_src_length: int = max_src_length

        self.embedding = nn.Embedding(self.output_size, self.embedding_size)
        self.attn = nn.Linear(self.embedding_size + self.decoder_hidden_size, max_src_length)
        self.attn_combine = nn.Linear(self.embedding_size + self.encoder_hidden_size, self.encoder_hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.encoder_hidden_size, self.decoder_hidden_size, num_hidden_layers)
        self.out = nn.Linear(self.decoder_hidden_size, self.output_size)

    def forward(self,
                input_tensor: torch.Tensor,
                hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                batch_size: int):

        if encoder_outputs.shape[0] != self.max_src_length:
            raise ValueError("Encoder outputs provided to this method must have same length as self.max_src_length:" +
                             f"\t{encoder_outputs.shape[0]} != {self.max_src_length}")

        # actual_src_length: int = max(self.max_src_length, input_tensor.shape[0])
        # print(f"self.max_src_length={self.max_src_length}\tinput_tensor.shape[0]={input_tensor.shape[0]}")
        verify_shape(tensor=input_tensor, expected=[1, batch_size])
        verify_shape(tensor=hidden, expected=[self.gru.num_layers, batch_size, self.gru.hidden_size])
        verify_shape(tensor=encoder_outputs, expected=[self.max_src_length, batch_size, self.encoder_hidden_size])

        # input_tensor.shape:    [1, 1]
        # hidden.shape:          [num_hidden_layers=1, batch_size=1, decoder_hidden_size]
        # encoder_outputs.shape: [src_seq_len, encoder_hidden_size]

        #if input_tensor.shape == torch.Size([]):
        #    raise RuntimeError(f"input_tensor.shape={input_tensor.shape} is a problem")

        # if self.embedding(input_tensor).shape != self.embedding(input_tensor).view(1, 1, -1).shape:
        #    raise RuntimeError(f"input_tensor.shape={input_tensor.shape}\tembedding is {self.embedding(input_tensor).shape} vs expected {self.embedding(input_tensor).view(1, 1, -1).shape}")

        # print(f"input_tensor={input_tensor}\tdecoder input_tensor.shape={input_tensor.shape}\t\t" +
        #      f"decoder hidden.shape={hidden.shape}\t\t" +
        #       f"encoder_outputs.shape={encoder_outputs.shape}") #\t\tembedded.shape={embedded.shape}")


        # TODO: It should be safe to remove .view(1, 1, -1), as it appears to be a noop
        embedded = self.embedding(input_tensor) #.view(1, 1, -1)

        verify_shape(tensor=embedded, expected=[1, batch_size, self.embedding_size])

        # self.embedding(input_tensor).shape:  [1, 1, decoder_embedding_size]
        # embedded.shape:                      [1, 1, decoder_embedding_size]

        # print(f"self.embedding(input_tensor).shape={self.embedding(input_tensor).shape}\t\t" +
        #      f"self.embedding(input_tensor).view(1, 1, -1).shape={self.embedding(input_tensor).view(1, 1, -1).shape}\t\t" +
        #      f"embedded.shape={embedded.shape}")

        embedded = self.dropout(embedded)

        verify_shape(tensor=embedded, expected=[1, batch_size, self.embedding_size])
        verify_shape(tensor=embedded[0], expected=[batch_size, self.embedding_size])
        verify_shape(tensor=hidden[-1], expected=[batch_size, self.gru.hidden_size])

        attn_input: torch.Tensor = torch.cat(tensors=(embedded[0], hidden[0]), dim=1)

        verify_shape(tensor=attn_input, expected=[batch_size, self.embedding_size + self.gru.hidden_size])

        # print(f"embedded[0].shape={embedded[0].shape}\t\t"+
        #      f"hidden[0].shape={hidden[0].shape}\t\t" )
        # sys.exit()
        #       f"torch.cat(tensors=(embedded[0], hidden[0]), dim=1).shape="+
        #       f"{torch.cat(tensors=(embedded[0], hidden[0]), dim=1).shape}")

        #print(f"self.attn(...).shape={self.attn(torch.cat(tensors=(embedded[0], hidden[0]), dim=1)).shape}\t\t"+
        #       f"softmax(...).shape="+
        #       f"{softmax(self.attn(torch.cat(tensors=(embedded[0], hidden[0]), dim=1)), dim=1).shape}")
        # embedded.shape:                      [1, 1, decoder_embedding_size]
        # embedded[0].shape:                      [1, decoder_embedding_size]
        #
        # hidden.shape:                        [1, 1, decoder_hidden_size]
        # hidden[0].shape:                        [1, decoder_hidden_size]
        #
        # torch.cat(tensors=(embedded[0], hidden[0]), dim=1).shape:  [1, embedded.shape[2]+hidden.shape[2]]
        #
        # self.attn(...).shape:                                      [1, decoder_max_len]
        # softmax(self.attn(...)).shape:                             [1, decoder_max_len]
        attn_weights = softmax(self.attn(attn_input), dim=1)

        verify_shape(tensor=attn_weights, expected=[batch_size, self.max_src_length])
        verify_shape(tensor=encoder_outputs, expected=[self.max_src_length, batch_size, self.encoder_hidden_size])

        # Permute dimensions to prepare for batched matrix-matrix multiply
        encoder_outputs = encoder_outputs.permute(1, 2, 0)
        attn_weights = attn_weights.unsqueeze(2)

        verify_shape(tensor=encoder_outputs, expected=[batch_size, self.encoder_hidden_size, self.max_src_length])
        verify_shape(tensor=attn_weights, expected=[batch_size, self.max_src_length, 1])

        #print(f"attn_weights.shape={attn_weights.shape}\t\t"+
        #       f"encoder_outputs.shape={encoder_outputs.shape}")


        #import sys;

        #sys.exit()

        # print(f"attn_weights.unsqueeze(0).shape={attn_weights.unsqueeze(0).shape}\t\t"+
        #       f"encoder_outputs.unsqueeze(0).shape={encoder_outputs.unsqueeze(0).shape}")

        # attn_weights.shape:                  [1, decoder_max_len]
        # encoder_outputs.shape:                  [decoder_max_len, encoder_hidden_size]
        #
        # attn_weights.unsqueeze(0).shape:     [1, 1, decoder_max_len]
        # encoder_outputs.unsqueeze(0).shape:     [1, decoder_max_len, encoder_hidden_size]
        #attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))  # <-- Original
        attn_applied = torch.bmm(encoder_outputs, attn_weights)   # <-- Batched

        # Get rid of superfluous final dimension
        #attn_applied = attn_applied.squeeze(dim=2)
        #verify_shape(tensor=attn_applied, expected=[batch_size, self.encoder_hidden_size])



        # print(f"attn_applied.shape={attn_applied.shape}\t\t"+
        #       f"embedded[0].shape={embedded[0].shape}\t\t"+
        #       f"attn_applied[0].shape={attn_applied[0].shape}\t\t"
        #       f"torch.cat(...).shape={torch.cat((embedded[0], attn_applied[0]), 1).shape}")

        # embedded.shape:                                  [1, batch_size=1, decoder_embedding_size]
        # attn_applied.shape:                              [1, batch_size=1, encoder_hidden_size]
        #
        # embedded[0].shape:                                  [batch_size=1, decoder_embedding_size]
        # attn_applied[0].shape:                              [batch_size=1, encoder_hidden_size]
        #
        # torch.cat((embedded[0], attn_applied[0]), 1).shape: [batch_size=1, decoder_embedding_size+encoder_hidden_size]
        #

        verify_shape(tensor=attn_applied, expected=[batch_size, self.encoder_hidden_size, 1])
        verify_shape(tensor=embedded, expected=[1, batch_size, self.embedding_size])

        # # The final dimension of attn_applied and the first dimension of embedded
        # #   represents seq_len, which is not needed at this point.
        # attn_applied = attn_applied.squeeze(dim=2)
        # embedded = embedded.squeeze(dim=0)
        #
        # verify_shape(tensor=attn_applied, expected=[batch_size, self.encoder_hidden_size])
        # verify_shape(tensor=embedded, expected=[batch_size, self.embedding_size])
        #
        # output = torch.cat((embedded, attn_applied), dim=1)
        # verify_shape(tensor=output, expected=[batch_size, self.embedding_size + self.encoder_hidden_size])

        attn_applied = attn_applied.permute(2, 0, 1)

        verify_shape(tensor=attn_applied, expected=[1, batch_size, self.encoder_hidden_size])
        verify_shape(tensor=embedded, expected=[1, batch_size, self.embedding_size])

        output = torch.cat(tensors=(embedded, attn_applied), dim=2)

        verify_shape(tensor=output, expected=[1, batch_size, self.embedding_size + self.encoder_hidden_size])

        # print(f"output.shape={output.shape}")

        # output.shape:                                      [batch_size=1, encoder_hidden_size+decoder_embedding_size]
        # self.attn_combine(output).shape:                   [batch_size=1, decoder_hidden_size]
        # self.attn_combine(output).unsqueeze(0): [seq_len=1, batch_size=1, decoder_hidden_size]
        #
        output = self.attn_combine(output) #.unsqueeze(0)

        verify_shape(tensor=output, expected=[1, batch_size, self.encoder_hidden_size])


        # print(f"output.shape={output.shape}")
        # print(f"relu(output).shape={relu(output).shape}\t\thidden.shape={hidden.shape}")

        # output.shape:                [seq_length=1, batch_size=1, decoder_hidden_size]
        # relu(...).shape:             [seq_length=1, batch_size=1, decoder_hidden_size]
        # hidden.shape:                [num_layers=1, batch_size=1, decoder_hidden_size]
        #
        output = relu(output)

        verify_shape(tensor=output, expected=[1, batch_size, self.encoder_hidden_size])

        output, hidden = self.gru(output, hidden)

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

    def decode_sequence(self,
                        encoder_outputs: torch.Tensor,
                        start_symbol: int,
                        max_length: int,
                        target_tensor: torch.Tensor = None):

        encoded_sequence_length: int = encoder_outputs.shape[0]
        batch_size: int = encoder_outputs.shape[1]
        encoder_hidden_size: int = encoder_outputs.shape[2]
        device = encoder_outputs.device

        decoder_input = torch.tensor(data=[[start_symbol] * batch_size],
                                     dtype=torch.long,
                                     device=device)

        decoder_hidden = self.init_hidden(batch_size=batch_size, device=device)

        verify_shape(tensor=decoder_input, expected=[1, batch_size])
        verify_shape(tensor=decoder_hidden, expected=[self.gru.num_layers, batch_size, self.gru.hidden_size])

        results: List[torch.Tensor] = list()

        for index in range(max_length):
            verify_shape(tensor=decoder_input, expected=[1, batch_size])
            verify_shape(tensor=decoder_hidden,
                         expected=[self.gru.num_layers, batch_size, self.gru.hidden_size])

            decoder_output, decoder_hidden, decoder_attention = self(
                decoder_input, decoder_hidden, encoder_outputs, batch_size)

            verify_shape(tensor=decoder_output, expected=[batch_size, self.output_size])
            verify_shape(tensor=decoder_hidden,
                         expected=[self.gru.num_layers, batch_size, self.gru.hidden_size])
            verify_shape(tensor=decoder_attention, expected=[batch_size, encoded_sequence_length])

            results.append(decoder_output)

            if target_tensor is None:
                _, top_i = decoder_output.topk(1)
                decoder_input = top_i.detach().permute(1, 0)
            else:
                # print(f"target_tensor.shape={target_tensor.shape}\tindex={index}\tmax_length={max_length}")
                decoder_input = target_tensor[index].unsqueeze(dim=0)

        return torch.stack(tensors=results, dim=0)