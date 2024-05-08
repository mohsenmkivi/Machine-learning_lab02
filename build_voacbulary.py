import collections
import os


def remove_punctuation(text):
    """Replace punctuation symbols with spaces."""
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punct:
        text = text.replace(p, " ")
    return text


def read_document(filename):
    """Read the file and returns a list of words."""
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    words = []
    text = remove_punctuation(text.lower())
    for w in text.split():
        if len(w) > 2:
            words.append(w)
    return words


def write_vocabulary(voc, filename, n):
    """Write the n most frequent words to a file."""
    f = open(filename, "w")
    for word, count in sorted(voc.most_common(n)):
        print(word, file=f)
    f.close()


# The script reads all the documents in the smalltrain directory, uses
# the to form a vocabulary, writes it to the 'vocabulary.txt' file.
voc = collections.Counter()
for f in os.listdir("smalltrain/pos"):
    voc.update(read_document("smalltrain/pos/" + f))
for f in os.listdir("smalltrain/neg"):
    voc.update(read_document("smalltrain/neg/" + f))
write_vocabulary(voc, "vocabulary.txt", 1000)
