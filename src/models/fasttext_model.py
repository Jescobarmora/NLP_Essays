import fasttext

def prepare_fasttext_data(X, y, filename):
    with open(filename, 'w') as f:
        for text, label in zip(X, y):
            f.write(f'__label__{label} {" ".join(text)}\n')

def train_fasttext_model(train_file):
    model = fasttext.train_supervised(train_file, lr=1.0, epoch=25, wordNgrams=2, verbose=2, minCount=1)
    return model