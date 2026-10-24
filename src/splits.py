"""
Training / validation / test splits for one run-to-failure record.

Every function returns a *role* for each snapshot: ``"train"``, ``"val"``,
``"test"`` or ``"gap"`` (left out). Following week 5.5, models are fitted on
``train``, settings are chosen on ``val``, and ``test`` is scored once at
the end. The three methods deliberately differ in how much the scored
snapshots resemble the training snapshots (week 5.6):

* `time_split` - chronological. No future data is used, but the model must
  extrapolate to the failure it never saw (pessimistic).
* `interleaved_split` - a repeating pattern, e.g. 6 train, 2 val, 2 test in
  every 10. Neighbours 10 minutes apart are almost identical, so this
  mostly measures interpolation (optimistic, leaky).
* `block_split` - random blocks of several hours with a gap at each change
  of role, the idea behind ``GroupShuffleSplit`` with each block as a group.

A fourth method, holding out whole bearings from other tests, needs no
function: give every snapshot of a bearing the same role.

Typical use::

    roles = time_split(len(hours), ratios=(0.6, 0.2, 0.2))
    train, val, test = (roles == r for r in ("train", "val", "test"))
"""
import numpy as np

ROLES = np.array(["train", "val", "test"])


def _counts(n, ratios):
    """Split `n` items into three whole numbers in proportion to `ratios`."""
    ratios = np.asarray(ratios, dtype=float) / np.sum(ratios)
    counts = np.floor(n * ratios).astype(int)
    counts[0] += n - counts.sum()          # rounding leftovers go to training
    return counts


def time_split(n, ratios=(0.6, 0.2, 0.2)):
    """Chronological split: earliest snapshots train, latest test.

    Parameters
    ----------
    n : int
        Number of snapshots, in time order.
    ratios : tuple of 3 floats, optional
        Train, validation and test fractions.

    Returns
    -------
    numpy.ndarray of str
        Role of each snapshot.
    """
    return np.repeat(ROLES, _counts(n, ratios))


def interleaved_split(n, ratios=(0.6, 0.2, 0.2), period=10):
    """Repeat the same train / val / test pattern every `period` snapshots.

    Parameters
    ----------
    n : int
        Number of snapshots.
    ratios : tuple of 3 floats, optional
        Train, validation and test fractions. ``period * ratio`` should be
        whole numbers, e.g. period 20 for (0.7, 0.15, 0.15).
    period : int, optional
        Length of the repeating pattern.

    Returns
    -------
    numpy.ndarray of str
        Role of each snapshot.
    """
    pattern = np.repeat(ROLES, np.round(np.asarray(ratios) * period).astype(int))
    return pattern[np.arange(n) % pattern.size]


def block_split(t, ratios=(0.6, 0.2, 0.2), block_hours=5.0, gap_hours=1.0, seed=0):
    """Give whole blocks of time a random role, with gaps between roles.

    Parameters
    ----------
    t : array_like
        Snapshot times in hours, increasing.
    ratios : tuple of 3 floats, optional
        Train, validation and test fractions of the blocks.
    block_hours : float, optional
        Length of each block.
    gap_hours : float, optional
        Snapshots this close to a change of role become ``"gap"``, so
        near-identical neighbours never sit on both sides.
    seed : int, optional
        Random seed; different seeds give different block assignments.

    Returns
    -------
    numpy.ndarray of str
        Role of each snapshot.
    """
    t = np.asarray(t, dtype=float)
    block = ((t - t[0]) // block_hours).astype(int)
    n_blocks = block.max() + 1
    shuffled = np.random.default_rng(seed).permutation(n_blocks)
    block_role = np.empty(n_blocks, dtype=object)
    block_role[shuffled] = np.repeat(ROLES, _counts(n_blocks, ratios))
    roles = block_role[block].astype("<U5")
    change = t[1:][roles[1:] != roles[:-1]]  # times where the role changes
    for c in change:
        roles[np.abs(t - c) < gap_hours] = "gap"
    return roles
