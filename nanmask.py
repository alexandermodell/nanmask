import numpy as np
from .utils import take, put, range_exclude

def mask(X, axis=0, return_mask=True):

    # if axis is a tuple or list, mask along multiple axes
    if isinstance(axis, (tuple, list)):
        msks = []
        # compute a mask for each axis
        for ax in axis:
            other_axes = range_exclude(X.ndim, ax)
            msk = ~np.isnan(X).all(axis=other_axes)
            msks.append(msk)
        # apply the masked for each axis
        X_masked = X
        for ax, msk in zip(axis, msks):
            X_masked = take(X_masked, msk, axis=ax)

        if return_mask:
            return X_masked, msks
        else:
            return X_masked

    # if axis is an integer, mask along a single axis
    else:
        other_axes = range_exclude(X.ndim, axis)
        msk = ~np.isnan(X).all(axis=other_axes)
        X_masked = take(X, msk, axis)
        if return_mask:
            return X_masked, msk
        else:
            return X_masked

def unmask(X, msk, axis=0, fill=np.nan):
    if isinstance(axis, int):
        axis = [axis]
    if isinstance(msk, np.ndarray):
        msk = [msk]

    # check if inds and axis have the same length
    if len(msk) != len(axis):
        raise ValueError("msk and axis must have the same length")

    shape = list(X.shape)
    for ax, m in zip(axis, msk):
        shape[ax] = len(m)
    X_unmasked = np.full(shape, fill)
    put(X_unmasked, X, msk, axis)
    return X_unmasked