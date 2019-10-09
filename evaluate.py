import argparse
from typing import List

import torch

from vocab import Vocabulary, VocabularyEntry, ReservedSymbols
from pg import Corpus
from seq2seq import EncoderRNN, AttnDecoderRNN


def evaluate(vocab: Vocabulary,
             corpus_filename: str,
             encoder: EncoderRNN,
             decoder: AttnDecoderRNN,
             max_tgt_length: int):

    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    decoder.to(device)
    
    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        corpus = Corpus(filename=corpus_filename,
                        max_src_length=decoder.max_src_length,
                        vocab=vocab,
                        device=device)
        
        for batch in torch.utils.data.DataLoader(dataset=corpus, batch_size=1):

            input_tensor: torch.Tensor = batch["data"].permute(1, 0)

            encoder_outputs = encoder.encode_sequence(input_tensor)

            decoder_output = decoder.decode_sequence(encoder_outputs=encoder_outputs,
                                                     start_symbol=corpus.characters.start_of_sequence.integer,
                                                     max_length=max_tgt_length)
            _, top_i = decoder_output.topk(k=1)

            predictions = top_i.squeeze(dim=2).squeeze(dim=1).tolist()

            predicted_string = "".join([corpus.characters[i].string for i in predictions])

            print(predicted_string)


def configure_evaluation(args: List[str]) -> argparse.Namespace:

    import configargparse

    p = configargparse.get_argument_parser()
    p.add('-c', '--config', required=True, is_config_file=True, help='configuration file')
    p.add('--vocab', required=True, help='Pickle file containing a Vocabulary object')
    p.add('--corpus', required=True, help='Filename of corpus to evaluate')
    p.add('--encoder', required=True, help='Pytorch save file containing a trained EncoderRNN object')
    p.add('--decoder', required=True, help='Pytorch save file containing a trained AttnDecoderRNN object')
    p.add('--max_decode_length', required=True, type=int, help='Maximum length string to generate during decoding')

    return p.parse_args(args=args)


if __name__ == "__main__":

    import pickle
    import sys

    options = configure_evaluation(sys.argv[1:])

    evaluate(vocab=pickle.load(open(options.vocab, "rb")),
             corpus_filename=options.corpus,
             encoder=torch.load(options.encoder),
             decoder=torch.load(options.decoder),
             max_tgt_length=options.max_decode_length)
