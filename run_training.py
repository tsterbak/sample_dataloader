from tqdm import tqdm
from src.dataloader import TrainDataLoader
from src.model import DummySimilarityModel
from src.tokenizer import SimpleTokenizer


# CONGIG
BATCH_SIZE = 16
MAX_LENGTH = 20
SHUFFLE = True
DATAPATH = "data/Stsbenchmark/stsbenchmark/sts-train.csv"


def load_data(path):
    sentences1 = []
    sentences2 = []
    scores = []
    with open(path, "r") as csv_file:
        for line in csv_file.readlines():
            row = line.split("\t")
            sentences1.append(row[5])
            sentences2.append(row[6])
            scores.append(row[4])
    return sentences1, sentences2, scores


if __name__ == "__main__":


    # Load the data set
    sentences1, sentences2, scores = load_data(DATAPATH)

    # prepare the tokenizer
    tokenizer = SimpleTokenizer(max_length=MAX_LENGTH)
    tokenizer.fit(sentences1)


    # setup the dataloader
    dl = TrainDataLoader(
        sentences1, sentences2, scores,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE
    )
    dl.process()

    # setup the model
    blackbox_model = DummySimilarityModel()

    # training
    for batch in tqdm(dl):
        blackbox_model.update(*batch)
