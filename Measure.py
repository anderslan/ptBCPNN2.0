import numpy as np

def binary_entropy(f):
    """Binary entropy function H(f) in bits."""
    if f == 0 or f == 1:
        return 0
    return -f * np.log2(f) - (1 - f) * np.log2(1 - f)

def bits_per_synapse(N, f, P, symmetric=True, precision=4):
    """
    Calculate bits of information stored per synapse in an associative memory.

    Parameters:
        N (int): Number of neurons.
        f (float): Sparsity (fraction of active units per pattern).
        P (int): Number of stored patterns.
        symmetric (bool): Whether weight matrix is symmetric (recurrent network).
        precision (int): Decimal precision of the output.

    Returns:
        dict: Contains entropy, total bits, free parameters, and bits per synapse.
    """
    H_f = binary_entropy(f)
    if symmetric:
        free_params = N * (N - 1) / 2  # no self-connections, symmetric
    else:
        free_params = N * N  # asymmetric, full matrix
    total_bits = P * N * H_f
    bits_per_syn = total_bits / free_params

    # return {
    #     'N': N,
    #     'f': f,
    #     'P': P,
    #     'H(f)': round(H_f, precision),
    #     'Free Parameters': int(free_params),
    #     'Total Bits Stored': round(total_bits, precision),
    #     'Bits per Synapse': round(bits_per_syn, precision)
    # }

    return round(bits_per_syn, precision)
