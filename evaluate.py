import torch

from vocab import Vocabulary, VocabularyEntry, ReservedSymbols
from pg import Corpus
from seq2seq import EncoderRNN, AttnDecoderRNN


def evaluate(vocab: Vocabulary,
             corpus_filename: str,
             encoder: EncoderRNN,
             decoder: AttnDecoderRNN):

    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    decoder.to(device)
    
    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        corpus = Corpus(name="eval",
                        filename=corpus_filename,
                        max_length=decoder.max_src_length,
                        vocab=vocab,
                        device=device)
        
        for batch in torch.utils.data.DataLoader(dataset=corpus, batch_size=1):

            input_tensor: torch.Tensor = batch["data"].permute(1, 0)

            encoder_outputs = encoder.encode_sequence(input_tensor)

            decoder_output = decoder.decode_sequence(encoder_outputs=encoder_outputs,
                                                     start_symbol=corpus.characters.start_of_sequence.integer,
                                                     max_length=corpus.label_tensor_length)
            _, top_i = decoder_output.topk(k=1)

            predictions = top_i.squeeze(dim=2).squeeze(dim=1).tolist()

            predicted_string = "".join([corpus.characters[i].string for i in predictions])

            print(predicted_string)


if __name__ == "__main__":

    import pickle

    evaluate(vocab=pickle.load(open("shakespeare.symbols.pkl", "rb")),
             corpus_filename="../pytorch_examples/data/shakespeare.tiny",
             encoder=torch.load("shakespeare.tiny.encoder.pt"),
             decoder=torch.load("shakespeare.tiny.decoder.pt"))
