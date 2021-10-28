import os
from io import open
import torch
import pandas as pd
import json

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        with open(os.path.join(path, 'tokenizer.json'), 'r') as f:
            tokenizer_config = json.load(f)
            self.dictionary.word2idx = tokenizer_config['model']['vocab']
            idx2word = {self.dictionary.word2idx[w]: w for w in self.dictionary.word2idx.keys()}
            self.dictionary.idx2word = [idx2word[i] for i in range(len(self.dictionary.word2idx))]
        self.train = self.tokenize(os.path.join(path, 'train.json'))
        self.valid = self.tokenize(os.path.join(path, 'val.json'))
        self.test = self.valid

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        df = pd.read_json(path, lines=True)

        idss = []
        for i in range(len(df)):
            words = df.iloc[i]['text'].split()
            ids = []
            for word in words:
                ids.append(self.dictionary.word2idx[word])
            idss.append(torch.tensor(ids).type(torch.int64))
        ids = torch.cat(idss)

        return ids
