import re
from collections import Counter


def build_vocab(captions, min_freq=3):
    counter = Counter()

    for caption in captions:
        # remove punctuation + lowercase
        tokens = re.sub(r"[^\w\s]", "", caption.lower()).split()
        counter.update(tokens)

    # special tokens
    vocab = {
        "<pad>": 0,
        "<start>": 1,
        "<end>": 2,
        "<unk>": 3
    }

    idx = 4

    # sorted for reproducibility
    for word, freq in sorted(counter.items()):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab
