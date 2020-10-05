# An example dataloader for sentence similarity training

This repository contains the code for an example text dataloader for sentence similarity training.


### Dataset preparation

The dataset should contain at least three columns. Two columns should contain the text of sentences to compare. The third column should contain the similarity scores for supervised training.
Load these columns to memory and pass them to the `TrainDataLoader` class.


### Run the code

```bash
python run_training.py
```
