from typing import Iterable, List, Mapping, NamedTuple, Union


class VocabularyEntry(NamedTuple):
    integer: int
    string: str


class CoreVocabulary(NamedTuple):
    pad: VocabularyEntry
    oov: VocabularyEntry
    start_of_sequence: VocabularyEntry
    end_of_sequence: VocabularyEntry


class Vocabulary(CoreVocabulary, Mapping[Union[int, str], VocabularyEntry]):

    def __new__(cls, *,
                pad: VocabularyEntry,
                oov: VocabularyEntry,
                start_of_sequence: VocabularyEntry,
                end_of_sequence: VocabularyEntry,
                entry_list: List[VocabularyEntry],
                entry_map: Mapping[Union[int, str], VocabularyEntry]):
        self = super(Vocabulary, cls).__new__(cls, pad, oov, start_of_sequence, end_of_sequence)
        self._entry_list = entry_list
        self._entry_map = entry_map
        return self

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

        entry_map = {0: p, pad: p,
                     1: o, oov: o,
                     2: s, start_of_sequence: s,
                     3: e, end_of_sequence: e}

        entry_list = [p, o, s, e]

        for s in strings:

            if s == pad or s == oov or s == start_of_sequence or s == end_of_sequence:
                raise ValueError(f"Strings must not contain reserved string:\t{s}")

            if s not in entry_map:
                v = VocabularyEntry(integer=len(entry_list), string=s)
                entry_map[v.integer] = v
                entry_map[v.string] = v
                entry_list.append(v)

        return Vocabulary(pad=p,
                          oov=o,
                          start_of_sequence=s,
                          end_of_sequence=e,
                          entry_list=entry_list,
                          entry_map=entry_map)


if __name__ == '__main__':

    vocab = Vocabulary.construct(pad='<pad>',
                                 oov='<oov>',
                                 start_of_sequence='<s>',
                                 end_of_sequence='</s>',
                                 strings=['a', 'b', 'oov', 'c'])

    print(vocab.oov in vocab)
    print(None in vocab)
    print(8 in vocab)
    print(len(vocab))
