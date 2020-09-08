def split_into_chunks(list, split_dims):
    """
Split a list into evenly sized chunks. The last chunk will be smaller if the original list length is not divisible by 'split_dims'.

    :param list: List to be split.
    :param split_dims: Length of each split chunk.
    """
    # For item i in a range that is a length of l,
    for i in range(0, len(list), split_dims):
        # Create an index range for l of n items:
        yield list[i:i + split_dims]
