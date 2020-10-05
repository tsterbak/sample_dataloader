"""Dummy model class."""
import time
from typing import List


class DummySimilarityModel(object):
    """Dummy model for similarity training. For illustration only."""

    def __init__(self):
        self.__dummy_score = 1.0

    def update(self, sentences1: List[int], sentences2: List[int], scores: List[int]):
        """Update the model parameters based on data."""
        time.sleep(0.1)

    def predict(self, txt1: str, txt2: str) -> float:
        """Get the similarity score between two pieces of text."""
        return self.__dummy_score
