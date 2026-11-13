"""Feature construction for calibrating uncertain temporal predictions."""

import numpy as np


def direction_features(forward, reverse):
    total = forward + reverse

    def signed_fraction(masses):
        return np.divide(masses[:, 1:], masses[:, :1], out=np.zeros_like(masses[:, 1:]),
                         where=masses[:, :1] > 0).clip(-1, 1)

    a, b, combined = map(signed_fraction, (forward, reverse, total))
    uncertain = (np.abs(combined) <= 0.4) | ((a < 0) != (b < 0))
    log_ratio = (np.log(np.maximum(forward[:, 0], 1e-300))
                 - np.log(np.maximum(reverse[:, 0], 1e-300))).clip(-12, 12)
    return a, b, combined, log_ratio, uncertain


def feature_columns(prepared, counts, parity, rows, bit):
    a, b, combined, log_ratio, _ = prepared
    x, y = a[rows, bit], b[rows, bit]
    occupancy = counts[rows].astype(np.float64)
    return np.column_stack((combined[rows], x, y, log_ratio[rows], parity[rows, bit],
                            occupancy, occupancy * x[:, None], occupancy * y[:, None]))


def dual_search_columns(narrow, wide, counts, rows, bit):
    a, b, combined, ratio, _ = narrow
    c, d, refined, refined_ratio, _ = wide
    occupancy = counts[rows].astype(np.float64)
    x = refined[rows, bit]
    delta = x - combined[rows, bit]
    return np.column_stack((
        combined[rows], refined[rows], a[rows, bit], b[rows, bit], c[rows, bit], d[rows, bit],
        ratio[rows], refined_ratio[rows], delta, occupancy, occupancy * x[:, None],
        occupancy * delta[:, None],
    ))
