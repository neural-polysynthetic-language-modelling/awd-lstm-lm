import os
import torch

import pickle
from collections import Counter


class Dictionary(object):
    """
    This class contains the mapping from words to indexes and from indexes
    to words. 
    
    In addition, for this version which uses the flattened TPR
    vectors, this class also contains a mapping from word to flattened TPR vector
    """
    def __init__(self, dict_path):
        """
        Args(dict_path): specifies the location of the pickled dictionary
                         for going from words to flattened TPR vectors
        """
        self.word2vec = {}
        self.word2idx = {}
        self.idx2word = []
        # dict_path should be pointing to a pickle file
        if os.path.isfile(dict_path):
            print(dict_path)
            with open(dict_path, "rb") as in_pkl:
                py_dict = pickle.load(in_pkl)
                self.word2vec = {word: torch.FloatTensor(py_dict[word]) for word in py_dict.keys()}

        else:
            raise Exception("The pickle file " + dict_path +
                            " specified is not present")

        self.emsize = len(list(self.word2vec.values())[0])
        self.counter = Counter()
        self.total = 0
        self.idx2word = [word for word in self.word2vec.keys()]
        self.word2idx = {word[1]:word[0] for word in enumerate(self.idx2word)}
        self.add_unk()

    #def add_space(self):
    #    """
    #    This adds an entry in the dictionary for a word separation
    #    apparently this isn't necessary because i'm not modelling word boundaries rn
    #    """
        
    def add_unk(self):
        """
        This adds an entry in the dictionary for unknown tokens
        this is the average value of each index in all flattened vectors

        There's probably a more optimal way to do this that generates a matrix
        and then adds the values in one go but this is likely fast enough
        """
        _ = self.add_word('<<unk>>')
        print(type(self.word2vec))
        for value in self.word2vec.values():
            self.word2vec["<<unk>>"] += value

    def add_word(self, word):
        self.idx2word.append(word)
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word) - 1
            embed_dim = self.emsize
            self.word2vec[word] = torch.zeros(embed_dim)

        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, dict_path, morph_sep=">"):
        print("Hello")
        self.dictionary = Dictionary(dict_path)
        print(self.dictionary)
        self.morph_sep = morph_sep
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Tokenize file content
        n_tokens = len(self.dictionary)
        print("n-tokens " + str(n_tokens))
        with open(path, 'r') as f:
            ids = []
            token = 0
            for line in f:
                print(token)
                words = line.split()# + ['<eos>']
                for word in words:
                    morphs = word.split(self.morph_sep)
                    for morph in morphs:
                        try:
                            ids.append(self.dictionary.word2idx[morph])
                        except KeyError:
                            #print(self.dictionary.word2idx["<<unk>>"])
                            ids.append(self.dictionary.word2idx["<<unk>>"])
                        token += 1

        return torch.LongTensor(ids)
