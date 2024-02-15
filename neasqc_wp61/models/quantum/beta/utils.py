import numpy as np 

def normalise_vector(vector : np.array) -> np.array:
    """
    Normalise a vector so that it has norm 1.

    Parameters
    ----------
    vector : np.array
        Vector to be normalised.

    Returns
    ------
    np.array
        Normalised vector.
    """
    return vector/np.linalg.norm(vector)

def pad_vector_with_zeros(vector : np.array) -> np.array:
    """
    Pad a vector with zeros so that length
    is a power of 2.

    Parameters
    ----------
    vector : np.array
        Vector to be padded.
    
    Returns
    -------
    np.array
        Padded vector.
    """
    n = len(vector)
    next_power_of_2 = 2 ** int(np.ceil(np.log2(n)))
    zero_padding = np.zeros(next_power_of_2 - n)
    return np.concatenate((vector, zero_padding), axis = 0)

