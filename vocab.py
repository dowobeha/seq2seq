from typing import Iterable, List, Mapping, MutableMapping, NamedTuple, Union


class VocabularyEntry(NamedTuple):
    integer: int
    string: str


class ReservedSymbols(NamedTuple):
    pad: VocabularyEntry
    oov: VocabularyEntry
    start_of_sequence: VocabularyEntry
    end_of_sequence: VocabularyEntry


class Vocabulary:

    def __init__(self, *,
                 reserved_symbols: ReservedSymbols,
                 entry_list: List[VocabularyEntry],
                 entry_map: Mapping[Union[int, str], VocabularyEntry]):
        self._reserved_symbols = reserved_symbols
        self._entry_list = entry_list
        self._entry_map = entry_map

    @property
    def oov(self) -> VocabularyEntry:
        return self._reserved_symbols.oov

    @property
    def pad(self) -> VocabularyEntry:
        return self._reserved_symbols.pad

    @property
    def start_of_sequence(self) -> VocabularyEntry:
        return self._reserved_symbols.start_of_sequence

    @property
    def end_of_sequence(self) -> VocabularyEntry:
        return self._reserved_symbols.end_of_sequence

    def __len__(self):
        return len(self._entry_list)

    def __getitem__(self, key: Union[int, str]) -> VocabularyEntry:
        if isinstance(key, int) or isinstance(key, str):
            if key in self._entry_map:
                return self._entry_map[key]
            else:
                return self.oov
        else:
            raise TypeError(f"Vocabulary key must be int or str, not {type(key)}")

    def __iter__(self):
        return iter(self._entry_list)

    def __contains__(self, item):
        if isinstance(item, int) or isinstance(item, str):
            return item in self._entry_map
        elif isinstance(item, VocabularyEntry):
            return item in self._entry_list
        else:
            return False

    @staticmethod
    def construct(*,
                  pad: str,
                  oov: str,
                  start_of_sequence: str,
                  end_of_sequence: str,
                  strings: Iterable[str]) -> "Vocabulary":

        p = VocabularyEntry(integer=0, string=pad)
        o = VocabularyEntry(integer=1, string=oov)
        s = VocabularyEntry(integer=2, string=start_of_sequence)
        e = VocabularyEntry(integer=3, string=end_of_sequence)

        entry_map: MutableMapping[Union[int, str], VocabularyEntry] = {0: p, pad: p,
                                                                       1: o, oov: o,
                                                                       2: s, start_of_sequence: s,
                                                                       3: e, end_of_sequence: e}

        entry_list = [p, o, s, e]

        for string in strings:

            if string == pad or string == oov or string == start_of_sequence or string == end_of_sequence:
                raise ValueError(f"Strings must not contain reserved string:\t{string}")

            if string not in entry_map:
                v = VocabularyEntry(integer=len(entry_list), string=string)
                entry_map[v.integer] = v
                entry_map[v.string] = v
                entry_list.append(v)

        reserved: ReservedSymbols = ReservedSymbols(pad=p, oov=o, start_of_sequence=s, end_of_sequence=e)

        return Vocabulary(reserved_symbols=reserved,
                          entry_list=entry_list,
                          entry_map=entry_map)


if __name__ == '__main__':

    import pickle
    import sys

    if len(sys.argv) != 3:
        print(f"Usage:\t{sys.argv[0]} one_symbol_per_line.txt vocab.pkl")
        sys.exit(-1)

    with open(sys.argv[1]) as symbol_file, open(sys.argv[2], 'wb') as pickle_file:

        print(f"Reading symbols from {sys.argv[1]}...", file=sys.stderr)
        symbols = [line.strip() for line in symbol_file]

        print(f"Constructing vocabulary from symbols...", file=sys.stderr)
        vocab = Vocabulary.construct(pad='<pad>',
                                     oov='<oov>',
                                     start_of_sequence='<s>',
                                     end_of_sequence='</s>',
                                     strings=symbols)

        print(f"Writing vocabulary of {len(vocab)} symbols to {sys.argv[2]}...", file=sys.stderr)
        pickle.dump(vocab, pickle_file)

        print(f"Completed writing vocabulary of {len(vocab)} symbols to {sys.argv[2]}", file=sys.stderr)
