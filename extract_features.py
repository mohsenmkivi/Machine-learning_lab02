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
for f in os.listdir("aclImdb/aclImdb/train/pos"):
    documents.append(read_document("aclImdb/aclImdb/train/pos/" + f, voc))
    labels.append(1)
for f in os.listdir("aclImdb/aclImdb/train/neg"):
    documents.append(read_document("aclImdb/aclImdb/train/neg/" + f, voc))
    labels.append(0)
# np.stack transforms the list of vectors into a 2D array.
X_train = np.stack(documents)
Y_train = np.array(labels)
# The following line append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt.
data = np.concatenate([X_train, Y_train[:, None]], 1)
np.savetxt("big_train.txt.gz", data)

documents = []
labels = []
for f in os.listdir("aclImdb/aclImdb/validation/pos"):
    documents.append(read_document("aclImdb/aclImdb/validation/pos/" + f, voc))
    labels.append(1)
for f in os.listdir("aclImdb/aclImdb/validation/neg"):
    documents.append(read_document("aclImdb/aclImdb/validation/neg/" + f, voc))
    labels.append(0)
# np.stack transforms the list of vectors into a 2D array.
X_validation = np.stack(documents)
Y_validation = np.array(labels)
# The following line append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt.
data = np.concatenate([X_validation, Y_validation[:, None]], 1)
np.savetxt("validation.txt.gz", data)

documents = []
labels = []
for f in os.listdir("aclImdb/aclImdb/test/pos"):
    documents.append(read_document("aclImdb/aclImdb/test/pos/" + f, voc))
    labels.append(1)
for f in os.listdir("aclImdb/aclImdb/test/neg"):
    documents.append(read_document("aclImdb/aclImdb/test/neg/" + f, voc))
    labels.append(0)
# np.stack transforms the list of vectors into a 2D array.
X_test = np.stack(documents)
Y_test = np.array(labels)
# The following line append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt.
data = np.concatenate([X_test, Y_test[:, None]], 1)
np.savetxt("test.txt.gz", data)
