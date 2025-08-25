def flatten(x):
    """
    Flattens a nested list or iterable into a single flat list.

    Parameters
    ----------
    x : iterable
        A (possibly nested) iterable of items.

    Returns
    -------
    list
        A flat list containing all the items from the nested structure.

    Examples
    --------
    >>> flatten([[1, [2, 3]], 4, [5, 6]])
    [1, 2, 3, 4, 5, 6]
    """
    return list(_flatten(x))

def _flatten(x):
    '''
    Generator that will flatten a list that may contain
    other lists (nested arbitrarily) and simple items
    into a flat list.

    >>> flat = flatten([[1,[2,3]],4,[5,6]])
    >>> list(flat)
    [1,2,3,4,5,6]

    '''
    for item in x:
        try:
            yield from flatten(item)
        except TypeError:
            yield item
