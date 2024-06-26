import numpy as np



def _check_size(spec, *xs):
    """Check that the arrays respect the specification.

    spec is a string of comma separated specifications.  Each
    specification apply to one element in xs.  The specification i a
    list of letters representing the expected number of dimensions (*
    if a scalar is expected).  If the same letter is used multiple
    times, the the dimensions must match.  A traling '?' makes None a
    valid value.

    """
    ss = list(map(str.strip, spec.split(",")))
    if len(ss) != len(xs):
        msg = "Not enough arguments (expected {}, got {})"
        raise ValueError(msg.format(len(ss), len(xs)))
    dims = {}
    for s, x in zip(ss, xs):
        if s.endswith("?"):
            if x is None:
                continue
            s = s[:-1]
        if s == "*":
            if np.isscalar(x):
                continue
            else:
                raise ValueError("Scalar value expected")
        if len(s) != np.ndim(x):
            msg = "Expected an array of {} dimensions ({} dimensions found)"
            raise ValueError(msg.format(len(s), np.ndim(x)))
        for n, d in enumerate(s):
            k = x.shape[n]
            if d not in dims:
                dims[d] = k
            elif k != dims[d]:
                msg = "Dimensions do not agree (got {} and {})"
                raise ValueError(msg.format(dims[d], k))


def _check_labels(Y, nclasses=None):
    """Check that data can represent class labels."""
    if not np.issubdtype(Y.dtype, np.integer):
        if np.abs(np.modf(Y)[0]).max() > 0:
            raise ValueError("Expected integers")
        Y = Y.astype(np.int32)
    if Y.min() < 0:
        raise ValueError("Labels cannot be negative")
    if nclasses is not None and Y.max() >= nclasses:
        msg = "Invalid labels (maximum is {}, got {})"
        raise ValueError(msg.format(nclasses - 1, Y.max()))
    return Y


def _check_categorical(X):
    """Check that X contain categorical data.

    If needed X is converted to int.
    """
    if not np.issubdtype(X.dtype, np.integer):
        if np.abs(np.modf(X)[0]).max() > 0:
            raise ValueError("Categorical data must be integers")
        X = X.astype(np.int32)
    if X.min() < 0:
        raise ValueError("Categorical data cannot be negative")
    return X
def svm_inference(X, w, b):
    """SVM prediction of the class labels.

    Parameters
    ----------
    X : ndarray, shape (m, n)
         input features (one row per feature vector).
    w : ndarray, shape (n,)
         weight vector.
    b : float
         scalar bias.

    Returns
    -------
    ndarray, shape (m,)
        predicted labels (one per feature vector).
    ndarray, shape (m,)
        classification scores (one per feature vector).
    """
    X = np.asarray(X)
    w = np.asarray(w)
    _check_size("mn, n, *", X, w, b)
    logits = X @ w + b
    labels = (logits > 0).astype(int)
    return labels, logits


def svm_train(X, Y, lambda_, lr=1e-3, steps=1000, init_w=None, init_b=0):
    """Train a binary SVM classifier.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate
    steps : int
        number of training steps
    init_w : ndarray, shape (n,)
        initial weights (None for zero initialization)
    init_b : float
        initial bias

    Returns
    -------
    w : ndarray, shape (n,)
        learned weight vector.
    b : float
        learned bias.
    """
    Y = np.asarray(Y).astype(int)
    X = np.asarray(X)
    if init_w is not None:
        init_w = np.asfarray(init_w)
    _check_size("mn, m, n?, *", X, Y, init_w, init_b)
    Y = _check_labels(Y, 2)
    m, n = X.shape
    w = (init_w if init_w is not None else np.zeros(n))
    b = init_b
    C = (2 * Y) - 1
    for step in range(steps):
        labels, logits = svm_inference(X, w, b)
        hinge_diff = -C * ((C * logits) < 1)
        grad_w = (hinge_diff @ X) / m + lambda_ * w
        grad_b = hinge_diff.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def hinge_loss(labels, logits):
    """Average hinge loss.

    Parameters
    ----------
    labels : ndarray, shape (m,)
        binary target labels (0 or 1).
    logits : ndarray, shape (m,)
        classification scores (logits).

    Returns
    -------
    float
        average hinge loss.
    """
    labels = np.asarray(labels).astype(int)
    logits = np.asarray(logits)
    _check_size("m, m", labels, logits)
    labels = _check_labels(labels, 2)
    loss = np.maximum(0, 1 - (2 * labels - 1) * logits)
    return loss.mean()
