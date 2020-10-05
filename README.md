# An example dataloader for sentence similarity training

This repository contains the code for an example text dataloader for sentence similarity training.


### Dataset preparation

The dataset should contain at least three columns. Two columns should contain the text of sentences to compare. The third column should contain the scores of similarity between the respective sentences for training.
Load these columns into memory, preapre a `SimpleTokenizer` object and pass them to the `TrainDataLoader` class. Then you can start iterating through the dataset in processed batches.

### Example usage

```python
dl = TrainDataLoader(
  sentences1,
  sentences2,
  scores,
  tokenizer=prepared_tokenizer,
  batch_size=16,
  shuffle=True
)

for batch in dl:
  # do something with the batch of data
```


### Run the code

```bash
python run_training.py
```

### Run tests

```bash
python -m pytest
```
