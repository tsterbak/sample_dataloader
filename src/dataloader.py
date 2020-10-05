"""Text dataloader for similarity training."""
import random
from typing import List
from src.tokenizer import SimpleTokenizer


class TrainDataLoader(object):

    def __init__(self, sentences1: List[str], sentences2: List[str], scores: List[float],
                 tokenizer: SimpleTokenizer, batch_size: int = 16, shuffle: bool = True):
        if len(sentences1) != len(sentences2):
            raise ValueError("Input length mismatch.")
        if len(sentences1) != len(scores):
            raise ValueError("Input length mismatch.")
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.tokenizer = tokenizer

    def process(self):
        """Tokenize and optional shuffle the data."""
        self.processed_sentences1 = [self.tokenizer.text_to_tokenids(s) for s in self.sentences1]
        self.processed_sentences2 = [self.tokenizer.text_to_tokenids(s) for s in self.sentences2]
        self.processed_scores = self.scores.copy()
        if self.shuffle:
            tmp = list(zip(self.processed_sentences1, self.processed_sentences2, self.processed_scores))
            random.shuffle(tmp)
            self.processed_sentences1, self.processed_sentences2, self.processed_scores = zip(*tmp)

    def __iter__(self):
        self.n_iter = 0
        return self

    def __len__(self):
        return len(self.sentences1) // self.batch_size

    def __next__(self):
        """Get the next batch of processed data.

        Drops the last incomplete batch."""
        if(self.n_iter >= len(self.processed_sentences1)//self.batch_size):
            raise StopIteration
        batch_start, batch_end = self.n_iter*self.batch_size, (self.n_iter+1)*self.batch_size
        batch = (
            self.processed_sentences1[batch_start:batch_end],
            self.processed_sentences2[batch_start:batch_end],
            self.processed_scores[batch_start:batch_end]
        )
        self.n_iter += 1
        return batch
