import numpy as np

def take(X, inds, axis=0):
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(inds, np.ndarray):
        inds = [inds]

    # check if inds and axis have the same length
    if len(inds) != len(axis):
        raise ValueError("inds and axis must have the same length")

    # constuct masks
    masks = [np.full(X.shape[i], True) for i in range(X.ndim)]    
    for ix, ax in zip(inds, axis):
        masks[ax] = ix

    return X[np.ix_(*masks)]

def put(X, Y, inds, axis=0, inplace=True):
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(inds, np.ndarray):
        inds = [inds]

    # check if inds and axis have the same length
    if len(inds) != len(axis):
        raise ValueError("inds and axis must have the same length")

    # constuct masks
    masks = [np.full(X.shape[i], True) for i in range(X.ndim)]    
    for ix, ax in zip(inds, axis):
        masks[ax] = ix

    if inplace:
        X[np.ix_(*masks)] = Y
    else:
        X_copy = X.copy()
        X_copy[np.ix_(*masks)] = Y
        return X_copy
    
def range_exclude(n, exclude):
    return tuple(i for i in range(n) if i != exclude)