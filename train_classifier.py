import numpy as np


def train_nb(X, Y):
    """Train a binary NB classifier."""
    # + 1 for the Laplacian smoothing
    pos_p = X[Y == 1, :].sum(0) + 1
    pos_p = pos_p / pos_p.sum()
    neg_p = X[Y == 0, :].sum(0) + 1
    neg_p = neg_p / neg_p.sum()
    w = np.log(pos_p) - np.log(neg_p)
    # Estimate P(0) and P(1) and compute b
    pos_prior = Y.mean()
    neg_prior = 1 - pos_prior
    b = np.log(pos_prior) - np.log(neg_prior)
    return w, b


def inference_nb(X, w, b):
    """Prediction of a binary NB classifier."""
    logits = X @ w + b
    return (logits > 0).astype(int)


# The script loads the training data and train a classifier.  It must
# be extended to evaluate the classifier on the test set.
data = np.loadtxt("train.txt.gz")
X = data[:, :-1]
Y = data[:, -1]
w, b = train_nb(X, Y)
predictions = inference_nb(X, w, b)
accuracy = (predictions == Y).mean()
print("Training accuracy:", accuracy * 100)


# This part detects the most relevant words for the classifier.
f = open("vocabulary.txt")
voc = f.read().split()
f.close()

indices = w.argsort()
print("NEGATIVE WORDS")
for i in indices[:20]:
    print(voc[i], w[i])

print()
print("POSITIVE WORDS")
for i in indices[-20:]:
    print(voc[i], w[i])

