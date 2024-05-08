import numpy as np
import os


def load_vocabulary(filename):
    """Load the vocabulary and returns it.

    The return value is a dictionary mapping words to numerical
indices.

    """
    f = open(filename)
    n = 0
    voc = {}
    for w in f.read().split():
        voc[w] = n
        n += 1
    f.close()
    return voc


def remove_punctuation(text):
    """Replace punctuation symbols with spaces."""
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punct:
        text = text.replace(p, " ")
    return text


def read_document(filename, voc):
    """Read a document and return its BoW representation."""
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    text = remove_punctuation(text.lower())
    # Start with all zeros
    bow = np.zeros(len(voc))
    for w in text.split():
        # If the word is the vocabulary...
        if w in voc:
            # ...increment the proper counter.
            index = voc[w]
            bow[index] += 1
    return bow


# The script compute the BoW representation of all the training
# documents.  This need to be extended to compute similar
# representations for the validation and the test set.
voc = load_vocabulary("vocabulary.txt")
documents = []
labels = []
for f in os.listdir("smalltrain/pos"):
    documents.append(read_document("smalltrain/pos/" + f, voc))
    labels.append(1)
for f in os.listdir("smalltrain/neg"):
    documents.append(read_document("smalltrain/neg/" + f, voc))
    labels.append(0)
# np.stack transforms the list of vectors into a 2D array.
X = np.stack(documents)
Y = np.array(labels)
# The following line append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt.
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("train.txt.gz", data)

