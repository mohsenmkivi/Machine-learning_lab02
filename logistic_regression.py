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

def log_nowarn(x):
    """Compute the logarithm without warnings in case of zeros."""
    with np.errstate(divide='ignore'):
        return np.log(x)


def logreg_inference(X, w, b):
    """Predict class probabilities.

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
        probability estimates (one per feature vector).
    """
    X = np.asarray(X)
    w = np.asarray(w)
    _check_size("mn, n, *", X, w, b)
    logits = X @ w + b
    return sigmoid(logits)


def binary_cross_entropy(Y, P):
    """Average cross entropy.

    Parameters
    ----------
    Y : ndarray, shape (m,)
        binary target labels (0 or 1).
    P : ndarray, shape (m,)
        probability estimates.

    Returns
    -------
    float
        average cross entropy.
    """
    Y = np.asarray(Y).astype(int)
    P = np.asarray(P)
    _check_size("m, m", Y, P)
    Y = _check_labels(Y, 2)
    log1 = log_nowarn(P)
    log0 = log_nowarn(1 - P)
    e = -log1[Y == 1].sum() - log0[Y == 0].sum()
    return e / Y.size


def logreg_train(X, Y, lambda_, lr=1e-3, steps=1000, init_w=None,
                 init_b=0):
    """Train a binary classifier based on L2-regularized logistic regression.

    Parameters
    ----------

    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate.
    steps : int
        number of training steps.
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
    for step in range(steps):
        P = logreg_inference(X, w, b)
        grad_w = ((P - Y) @ X) / m + 2 * lambda_ * w
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def logreg_l1_train(X, Y, lambda_, lr=1e-3, steps=1000, init_w=None, init_b=0):
    """Train a binary classifier based on L1-regularized logistic regression.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        training features.
    Y : ndarray, shape (m,)
        binary training labels.
    lambda_ : float
        regularization coefficient.
    lr : float
        learning rate.
    steps : int
        number of training steps.
    loss : ndarray, shape (steps,)
        loss value after each training step.

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
    for step in range(steps):
        P = logreg_inference(X, w, b)
        grad_w = ((P - Y) @ X) / m + lambda_ * np.sign(w)
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def sigmoid(z):
    """Elementwise sigmoid function.

    Parameters
    ----------
    z : ndarray
         input

    Returns
    -------
    ndarray, (same shape of z)
        the sigmoid of z
    """
    z = np.asarray(z)
    return 1 / (1 + np.exp(-z))
