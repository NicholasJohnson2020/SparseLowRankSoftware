function generate_synthetic_data(n, param_frac, sigma; signal_to_noise=10,
    fixed_rank_sparse=false, fixed_rank=1, fixed_sparse=0)

    if fixed_rank_sparse
        k_sparse = fixed_sparse
        k_rank = fixed_rank
    else
        k_sparse = Int64(floor(param_frac * n^2))
        k_rank = Int64(floor(param_frac * n))
    end

    d = Normal(0, signal_to_noise * sigma / n^0.5)
    X = rand(d, (n, k_rank))
    Y = rand(d, (n, k_rank))
    L_0 = X * Y'

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

    d = Normal(0, sigma^2)
    noise = rand(d, (n, n))
    data = data + noise

    return data, L_0, S_0, k_sparse, k_rank

end;

function generate_synthetic_data_symmetric(n, param_frac, sigma;
    signal_to_noise=10, fixed_rank_sparse=false, fixed_rank=1, fixed_sparse=0)

    if fixed_rank_sparse
        k_sparse = fixed_sparse
        k_rank = fixed_rank
    else
        k_sparse = Int64(floor(param_frac * n^2))
        k_rank = Int64(floor(param_frac * n))
    end

    d = Normal(0, signal_to_noise * sigma / n^0.5)
    X = rand(d, (n, k_rank))
    L_0 = X * X'

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
