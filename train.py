import random
import sys
import time
from typing import List

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer

from pg import Corpus
from seq2seq import EncoderRNN, AttnDecoderRNN
from utils import time_since, verify_shape
from vocab import Vocabulary, VocabularyEntry, ReservedSymbols

def train(*,
          input_tensor: torch.Tensor,  # shape: [src_seq_len, batch_size]
          target_tensor: torch.Tensor,  # shape: [tgt_seq_len, batch_size]
          encoder: EncoderRNN,
          decoder: AttnDecoderRNN,
          encoder_optimizer: Optimizer,
          decoder_optimizer: Optimizer,
          criterion: nn.Module,
          device: torch.device,
          max_src_length: int,
          max_tgt_length: int,
          batch_size: int,
          start_of_sequence_symbol: int,
          teacher_forcing_ratio: float) -> float:
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss: torch.Tensor = torch.tensor(0, dtype=torch.float, device=device)  # shape: [] meaning this is a scalar

    encoder_outputs = encoder.encode_sequence(input_tensor)

    decoder_input = target_tensor[0].unsqueeze(dim=0)
    decoder_hidden = decoder.init_hidden(batch_size=batch_size, device=device)

    verify_shape(tensor=decoder_input, expected=[1, batch_size])
    verify_shape(tensor=target_tensor, expected=[max_tgt_length, batch_size])
    verify_shape(tensor=decoder_hidden, expected=[decoder.gru.num_layers, batch_size, decoder.gru.hidden_size])

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # use_teacher_forcing = False

    decoder_output = decoder.decode_sequence(encoder_outputs=encoder_outputs,
                                             start_symbol=start_of_sequence_symbol,
                                             max_length=max_tgt_length,
                                             target_tensor=target_tensor if use_teacher_forcing else None)
    # print(f"input_tensor.shape={input_tensor.shape}\tdecoder_output.shape={decoder_output.shape}\ttarget_tensor.shape={target_tensor.shape}\tmax_tgt_length={max_tgt_length}")

    # Our loss function requires predictions to be of the shape NxC, where N is the number of predictions and C is the number of possible predicted categories
    predictions = decoder_output.reshape(-1,
                                         decoder.output_size)  # Reshaping from [seq_len, batch_size, decoder.output_size] to [seq_len*batch_size, decoder.output_size]
    labels = target_tensor.reshape(
        -1)  # Reshaping from [seq_len, batch_size]                      to [seq_len*batch_size]
    loss += criterion(predictions, labels)
    # print(f"\t{decoder_output.view(-1,decoder_output.shape[-1]).shape}")
    # print(target_tensor.reshape(-1))
    #    print(f"\t{target_tensor.view(-1)}")
    # sys.exit()
    # loss += criterion(decoder_output.view(1,1,-1), target_tensor.view(-1))
    # loss += criterion(decoder_output.squeeze(dim=1), target_tensor.squeeze(dim=1))
    # for index, decoder_output in enumerate(start=1,
    #                                        iterable=decoder.decode_sequence(encoder_outputs=encoder_outputs,
    #                                               start_of_sequence_symbol=start_of_sequence_symbol,
    #                                               max_length=max_tgt_length,
    #                                               target_tensor=target_tensor if use_teacher_forcing else None)):
    #
    #     loss += criterion(decoder_output, target_tensor[index])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item()


def train_iters(*,  # data: Data,
                corpus: Corpus,
                encoder: EncoderRNN,
                decoder: AttnDecoderRNN,
                device: torch.device,
                n_iters: int,
                batch_size: int,
                teacher_forcing_ratio: float,
                print_every: int = 1000,
                learning_rate: float = 0.01
                ) -> None:
    data = torch.utils.data.DataLoader(dataset=corpus, batch_size=batch_size)

    start: float = time.time()
    plot_losses: List[float] = []
    print_loss_total: float = 0  # Reset every print_every
    plot_loss_total: float = 0  # Reset every plot_every

    encoder_optimizer: Optimizer = SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer: Optimizer = SGD(decoder.parameters(), lr=learning_rate)
    #
    # training_pairs: List[ParallelTensor] = [random.choice(data.pairs).tensors(source_vocab=data.source_vocab,
    #                                                                           target_vocab=data.target_vocab,
    #                                                                           device=device)
    #                                         for _ in range(n_iters)]

    criterion: nn.NLLLoss = nn.NLLLoss(reduction='mean')  # ignore_index=corpus.characters.pad_int)

    # for pair in parallel_data:
    #    print(f"src={len(pair['data'])}\ttgt={len(pair['labels'])}")

    for iteration in range(1, n_iters + 1):  # type: int

        # training_pair: ParallelTensor = training_pairs[iteration - 1]
        # input_tensor: torch.Tensor = training_pair.source   # shape: [seq_len, batch_size=1]
        # target_tensor: torch.Tensor = training_pair.target  # shape: [seq_len, batch_size=1]

        for batch in data:
            # print(f"batch['data'].shape={batch['data'].shape}\tbatch['labels'].shape{batch['labels'].shape}")
            # sys.exit()
            input_tensor: torch.Tensor = batch["data"].permute(1, 0)
            target_tensor: torch.Tensor = batch["labels"].permute(1, 0)

            actual_batch_size: int = min(batch_size, input_tensor.shape[1])

            verify_shape(tensor=input_tensor, expected=[corpus.word_tensor_length, actual_batch_size])
            verify_shape(tensor=target_tensor, expected=[corpus.label_tensor_length, actual_batch_size])

            # print(f"input_tensor.shape={input_tensor.shape}\t\ttarget_tensor.shape={target_tensor.shape}")
            # sys.exit()

            loss: float = train(input_tensor=input_tensor,
                                target_tensor=target_tensor,
                                encoder=encoder,
                                decoder=decoder,
                                encoder_optimizer=encoder_optimizer,
                                decoder_optimizer=decoder_optimizer,
                                criterion=criterion,
                                device=device,
                                max_src_length=corpus.word_tensor_length,
                                max_tgt_length=corpus.label_tensor_length,
                                batch_size=actual_batch_size,
                                start_of_sequence_symbol=corpus.characters.start_of_sequence.integer,
                                teacher_forcing_ratio=teacher_forcing_ratio)

            print_loss_total += loss
            plot_loss_total += loss

        if iteration % print_every == 0:
            print_loss_avg: float = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(since=start, percent=iteration / n_iters),
                                         iteration, iteration / n_iters * 100, print_loss_avg))
            sys.stdout.flush()


def run_training(*,
                 vocab_pickle_filename: str,
                 training_filename: str,
                 encoder_savefilename: str,
                 decoder_savefilename: str) -> None:

    import pickle

    vocab: Vocabulary = pickle.load(open(vocab_pickle_filename, "rb"))

    max_length = 10

    teacher_forcing_ratio = 0.5

    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    training_corpus = Corpus(name="training",
                             vocab=vocab,
                             filename=training_filename,
                             max_length=max_length,
                             device=device)

    # for word in test_corpus.words:
    #    print(f"{''.join(word.characters)}\t{''.join(word.label)}")
    # sys.exit()

    encoder1: EncoderRNN = EncoderRNN(input_size=len(training_corpus.characters),
                                      embedding_size=200,
                                      hidden_size=256,
                                      num_hidden_layers=1).to(device=device)

    attn_decoder1 = AttnDecoderRNN(embedding_size=190,
                                   decoder_hidden_size=7,
                                   encoder_hidden_size=encoder1.hidden_size,
                                   num_hidden_layers=1,
                                   output_size=len(training_corpus.characters),
                                   dropout_p=0.1,
                                   max_src_length=training_corpus.word_tensor_length).to(device=device)

    train_iters(corpus=training_corpus,
                encoder=encoder1,
                decoder=attn_decoder1,
                device=device,
                n_iters=10,
                batch_size=25000,
                print_every=1,
                learning_rate=0.01,
                teacher_forcing_ratio=teacher_forcing_ratio)

    print(f"Saving encoder to {encoder_savefilename}...")
    torch.save(encoder1, encoder_savefilename)

    print(f"Saving decoder to {decoder_savefilename}...")
    torch.save(attn_decoder1, decoder_savefilename)

    
if __name__ == "__main__":

    run_training(vocab_pickle_filename="shakespeare.symbols.pkl",
                 training_filename="../pytorch_examples/data/shakespeare.tiny",
                 encoder_savefilename="shakespeare.tiny.encoder.pt",
                 decoder_savefilename="shakespeare.tiny.decoder.pt")
