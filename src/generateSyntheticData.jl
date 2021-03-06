function generate_synthetic_data(n, param_frac, sigma; signal_to_noise=10,
    fixed_rank_sparse=false, fixed_rank=1, fixed_sparse=0)
    """
    This function samples synthetic data by adding noise to the sum of a
    random low rank matrix and a random sparse matrix.

    :param n: The row and column dimension of the data to be sampled (Int64).
    :param param_frac: Parameter that controls the rank and sparsity of the
                       sampled low rank and sparse matrix when
                       fixed_rank_sparse=false (Float64).
    :param sigma: Parameter that controls the absolute magnitude of the noise
                  (Float64).
    :param signal_to_noise: Parameter that controls the signal to noise ratio of
                            the sampled data (Float64).
    :param fixed_rank_sparse: If true, use fixed_rank and fixed_sparse as the
                              rank and sparsity respectively of the low rank
                              matrix and the sparse matrix to be sampled (Bool).
    :param fixed_rank: Rank of the low rank matrix to be sampled when
                       fixed_rank_sparse=true.
    :param fixed_sparse: Sparsity of the sparse matrix to be sampled when
                         fixed_rank_sparse=true

    :return: This function returns 5 values:
             1) The sampled n x n data matrix.
             2) The sampled n x n low rank matrix.
             3) The sampled n x n sparse matrix.
             4) The sparsity level of the sparse matrix.
             5) The rank of the low rank matrix.
    """

    if fixed_rank_sparse
        k_sparse = fixed_sparse
        k_rank = fixed_rank
    else
        k_sparse = Int64(floor(param_frac * n^2))
        k_rank = Int64(floor(param_frac * n))
    end

    # Sample a rank k_rank matrix
    d = Normal(0, signal_to_noise * sigma / n^0.5)
    X = rand(d, (n, k_rank))
    Y = rand(d, (n, k_rank))
    L_0 = X * Y'

    # Sample a sparsity k_sparse matrix
    d = Uniform(-5, 5)
    E = rand(d, (n, n))
    S_0 = zeros(n, n)

    indexes = []
    for i = 1:n
        for j = 1:n
            append!(indexes, [(i, j)])
        end
    end

    select_indexes = sample(indexes, k_sparse, replace=false)
    for index in select_indexes
        S_0[index[1], index[2]] = E[index[1], index[2]]
    end

    data = L_0 + S_0

    # Add noise to the sum of the sampled low rank and sparse matrices
    d = Normal(0, sigma^2)
    noise = rand(d, (n, n))
    data = data + noise

    return data, L_0, S_0, k_sparse, k_rank

end;

function generate_synthetic_data_symmetric(n, param_frac, sigma;
    signal_to_noise=10, fixed_rank_sparse=false, fixed_rank=1, fixed_sparse=0)
    """
    This function samples symmetric synthetic data by adding symmetric noise to
    the sum of a random symmetric low rank matrix and a random symmetric sparse
    matrix.

    :param n: The row and column dimension of the data to be sampled (Int64).
    :param param_frac: Parameter that controls the rank and sparsity of the
                       sampled low rank and sparse matrix when
                       fixed_rank_sparse=false (Float64).
    :param sigma: Parameter that controls the absolute magnitude of the noise
                  (Float64).
    :param signal_to_noise: Parameter that controls the signal to noise ratio of
                            the sampled data (Float64).
    :param fixed_rank_sparse: If true, use fixed_rank and fixed_sparse as the
                              rank and sparsity respectively of the low rank
                              matrix and the sparse matrix to be sampled (Bool).
    :param fixed_rank: Rank of the low rank matrix to be sampled when
                       fixed_rank_sparse=true.
    :param fixed_sparse: Sparsity of the sparse matrix to be sampled when
                         fixed_rank_sparse=true

    :return: This function returns 5 values:
             1) The sampled n-by-n data matrix.
             2) The sampled n-by-n low rank matrix.
             3) The sampled n-by-n sparse matrix.
             4) The sparsity level of the sparse matrix.
             5) The rank of the low rank matrix.
    """

    if fixed_rank_sparse
        k_sparse = fixed_sparse
        k_rank = fixed_rank
    else
        k_sparse = Int64(floor(param_frac * n^2))
        k_rank = Int64(floor(param_frac * n))
    end

    # Sample a rank k_rank symmetric matrix
    d = Normal(0, signal_to_noise * sigma / n^0.5)
    X = rand(d, (n, k_rank))
    L_0 = X * X'

    # Sample a sparsity k_sparse symmetric matrix
    d = Uniform(-5, 5)
    E = rand(d, (n, n))
    S_0 = zeros(n, n)

    indexes = []
    for i = 1:(n-1)
        for j = (i+1):n
            append!(indexes, [(i, j)])
        end
    end

    select_indexes = sample(indexes, Int64(floor(k_sparse/2)), replace=false)
    for index in select_indexes
        S_0[index[1], index[2]] = E[index[1], index[2]]
        S_0[index[2], index[1]] = S_0[index[1], index[2]]
    end

    if k_sparse % 2 == 1
        index = sample(1:n, 1)
        S_0[index, index] = E[index, index]
    end

    data = L_0 + S_0

    # Add symmetric noise to the sum of the sampled low rank and sparse matrices
    d = Normal(0, sigma^2)
    noise = rand(d, (n, n))
    for i = 1:(n-1)
        for j = (i+1):n
            noise[j, i] = noise[i, j]
        end
    end
    data = data + noise

    return data, L_0, S_0, k_sparse, k_rank

end;
