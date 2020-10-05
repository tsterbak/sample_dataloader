"""Simple whitespace tokenizer class to split up text."""
from typing import List


class SimpleTokenizer(object):
    """Simple whitespace tokenizer class.

    Split the input text by whitespace characters.

    :params:
        :max_length: Maximum length of the processed text. Pad or truncate accordingly.
        :padid:      Token id of the padding token.
        :unknownid:  Token id of the unknown word token.
    """

    def __init__(self, max_length: int = 20, padid: int = 0, unknownid: int = 1):
        self.max_length = max_length
        self.unknownid = unknownid
        self.padid = padid
        self._vocabulary = dict()

    def fit(self, texts: List[str]):
        """Fit the tokenizer.

        Split the text on whitespace and create vocabulary."""
        self._vocabulary["PAD"] = self.padid
        self._vocabulary["UNK"] = self.unknownid

        for text in texts:
            for token in text.split(" "):
                if not self._vocabulary.get(token):
                    self._vocabulary[token] = len(self._vocabulary)

    def text_to_tokenids(self, text: str) -> List[int]:
        """Process the input text and return a padded list of token ids."""
        return [
            self._vocabulary.get(token, self._vocabulary["UNK"])
            for token in text.split(" ")[:self.max_length]
        ] + [self._vocabulary["PAD"] for _ in range(0, self.max_length - len(text.split(" ")))]
