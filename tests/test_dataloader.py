import random
import pytest
from typing import List
from src.dataloader import TrainDataLoader


class DummyTokenizer():

    def text_to_tokenids(self, text: str) -> List[int]:
        return [random.randint(0, 100) for _ in text.split(" ")]


def test_batching():
    test_sentences1 = ["This is a test", "I am a test sentence.", "Test test test!"]
    test_sentences2 = ["This is another test", "I am a test sentence!", "Test test test!"]
    scores = [1, 1, 0]

    tokenizer=DummyTokenizer()

    dl = TrainDataLoader(
        test_sentences1,
        test_sentences2,
        scores,
        tokenizer=tokenizer,
        batch_size=2,
        shuffle=False
    )

    dl.process()

    dl_iter = iter(dl)

    batch1 = next(dl_iter)
    assert len(batch1[0]) == 2

    with pytest.raises(StopIteration):
        batch2 = next(dl_iter)
