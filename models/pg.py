from typing import List, Mapping

import torch
from torch.utils.data.dataset import Dataset

from utils import normalize_string
from vocab import Vocabulary


class Word:
    def __init__(self, characters: List[str]):
        self.characters: List[str] = characters
        self.label: List[str] = Word.generate_label(characters)

    def __str__(self) -> str:
        return f"{''.join(self.characters)}\t{''.join(self.label)}"

    @staticmethod
    def is_completely_alphabetic(characters: List[str]) -> bool:
        from functools import reduce
        return reduce(lambda a, b: a & b, [c.isalpha() for c in characters])

    @staticmethod
    def position_of_first_vowel(characters: List[str]) -> int:
        for position in range(len(characters)):
            char = characters[position]
            if char in "aeiouAEIOU":
                return position
        return len(characters)

    @staticmethod
    def generate_label(characters: List[str]) -> List[str]:
        if Word.is_completely_alphabetic(characters):
            first_vowel = Word.position_of_first_vowel(characters)
            prefix = characters[0:first_vowel]
            suffix = characters[first_vowel:]
            return suffix + ['-'] + prefix + ['a', 'y']
        else:
            return characters


class Corpus(Dataset):

    def __init__(self, *,
                 filename: str,
                 max_src_length: int,
                 device: torch.device,
                 vocab: Vocabulary = None):

        self.device = device

        self.words: List[Word] = Corpus.read_words(filename, max_src_length)

        if vocab is None:
            strings = list()
            for word in self.words:
                for character in word.label:
                    strings.append(character)
            self.characters: Vocabulary = Vocabulary.construct(pad="<pad>",
                                                               oov="<oov>",
                                                               start_of_sequence="<s>",
                                                               end_of_sequence="</s>",
                                                               strings=strings)
        else:
            self.characters: Vocabulary = vocab

        # self._max_word_length = max_src_length
        # self._max_label_length = max_tgt_length

        self._max_word_length = Corpus.calculate_longest([word.characters for word in self.words]) # ; print(f"{self._max_word_length} {max_src_length}")
        self._max_label_length = Corpus.calculate_longest([word.label for word in self.words]) # ;  print(f"{self._max_label_length} {max_tgt_length}")

        self.word_tensor_length = self._max_word_length + 2
        self.label_tensor_length = self._max_label_length + 1

    def __len__(self) -> int:
        return len(self.words)

    def __getitem__(self, index: int) -> Mapping[str, torch.LongTensor]:

        return {"data": self.create_tensor(sequence=self.words[index].characters,
                                           pad_to_length=self._max_word_length,
                                           include_start_of_sequence=True),

                "labels": self.create_tensor(sequence=self.words[index].label,
                                             pad_to_length=self._max_label_length,
                                             include_start_of_sequence=False),

                "start-of-sequence": torch.tensor(data=[self.characters.start_of_sequence.integer],
                                                  dtype=torch.long,
                                                  device=self.device)}

    def create_tensor(self, *, sequence: List[str], pad_to_length: int,
                      include_start_of_sequence: bool) -> torch.LongTensor:

        start_of_sequence: List[int] = [self.characters.start_of_sequence.integer] if include_start_of_sequence else []
        int_sequence: List[int] = [self.characters[s].integer for s in sequence]  # type: ignore[index]
        pads: List[int] = [self.characters.pad.integer] * max(0, (pad_to_length - len(sequence)))
        end_of_sequence: List[int] = [self.characters.end_of_sequence.integer]

        result = torch.tensor(data=start_of_sequence + int_sequence + end_of_sequence + pads,
                              dtype=torch.long,
                              device=self.device) 
        
        return result  # type: ignore[return-value]

    @staticmethod
    def read_words(filename: str, max_length: int) -> List[Word]:
        words: List[Word] = list()
        with open(filename, mode='rt', encoding='utf8') as f:
            for line in f:  # type: str
                for word in normalize_string(line).strip().split():  # type: str
                    if len(word) <= max_length:
                        words.append(Word(list(word)))
        return words

    @staticmethod
    def calculate_longest(sequences: List[List[str]]) -> int:
        longest: int = 0
        for sequence in sequences:
            length = len(sequence)
            if length > longest:
                longest = length
        return longest
