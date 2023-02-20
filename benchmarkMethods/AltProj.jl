function hard_threshold(A, threshold_param)
    """
    This function perfoms hard threholding of the matrix A with respect to the
    float threshold_param.

    :param A: An arbitrary n-by-n matrix.
    :param threshold_param: The desired hard thresholding value (Float64).

    :return: The n-by-n hard thresholding matrix of A.
    """
    filter_mat = abs.(A) .>= threshold_param
    return A .* filter_mat

end

function alternating_projection(A, k_rank; beta_mult=1, zeta_mult=1,
                                epsilon=1e-3)
    """
    This function implements the alternating projection algorithm to solve
    sparse plus low rank matrix decomposition.

    :param A: An arbitrary n-by-n matrix.
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param epsilon: A termination parameter for the algorithm (Float64).

    :return: This function returns two values. The first value is a tuple of two
             n-by-n arrays that correspond to the best feasible solution found
             by this method (the first element in the tuple is the matrix X and
             the second element is the matrix Y). The second value is the
             objective value (Float64) achieved by the returned feasible
             solution.
    """
    n = size(A)[1]

    U, sigma, Vt = svd(A)

    # Compute "empirical" incoherence parameter
    incoherence_param = 0
    for i=1:n

        incoherence_param = maximum([norm(U[i, :]) * sqrt(n / k_rank),
                                     norm(Vt[:, i]) * sqrt(n / k_rank),
                                     incoherence_param])

    end

    # Initialize algorithm
    beta = beta_mult * 4 * incoherence_param ^ 2 * k_rank / n
    current_threshold = beta * sigma[1]
    X_iterate = zeros(n, n)
    Y_iterate = hard_threshold(A - X_iterate, current_threshold)

    # Main loop
    for current_rank=1:k_rank

        T = 10 * log(n * beta * norm(A-Y_iterate) / epsilon)

        for iteration=1:(T+1)

            U, sigma, Vt = svd(A - Y_iterate)

            #current_threshold = beta * (sigma[current_rank + 1] +
            #                    0.5 ^ (iteration - 1) * sigma[current_rank])

            # Update sparse matrix threshold
            zeta = zeta_mult * incoherence_param * sigma[current_rank + 1] / sqrt(n)

            sigma[(current_rank + 1):end] .= 0

            # Perform projections
            X_iterate = U * Diagonal(sigma) * Vt
            Y_iterate = hard_threshold(A - X_iterate, zeta)
        end

        if norm(A - X_iterate - Y_iterate) <= epsilon
            return (X_iterate, Y_iterate)
        end
    end

    return (X_iterate, Y_iterate)

end

function cross_validate_AltProj(U, k_rank;
                                num_samples=30, train_frac=0.7,
                                candidate_betas=[10, 1, 0.1, 0.01],
                                candidate_zetas=[10, 1, 0.1, 0.01])
    """
    This function performs cross validation to select the regularization
    parameters lambda and mu for the optimization problem given by
        min ||U - X - Y||_F^2 + lambda * ||X||_F^2 + mu * ||Y||_F^2
        subject to rank(X) <= k_rank, ||Y||_0 <= k_sparse

    :param U: An arbitrary n-by-n matrix.
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param num_samples: The number of samples to draw when
                        performing cross validation (Int64).
    :param train_frac: The fraction of the data to be used as training data when
                       performing cross validation.
    :param candidate_gammas: List of values of gamma (Float64) to be
                              considered during cross validation.

    :return: This function returns 2 values.
             1) The best performing gamma value (Float64).
             2) A dictionary of cross validation scores of all parameters.
    """

    n = size(U)[1]
    val_dim = Int(floor(n * (1-sqrt(train_frac))))
    train_dim = n - val_dim

    param_scores = Dict()
    for beta in candidate_betas, zeta in candidate_zetas
        param_scores[(beta, zeta)] = 0
    end

    for trial=1:num_samples

        permutation = randperm(n)
        val_indices = permutation[1:val_dim]
        train_indices = permutation[(val_dim+1):end]

        val_data = U[val_indices, val_indices]
        train_data = U[train_indices, train_indices]

        LL_block_data = U[val_indices, train_indices]
        UR_block_data = U[train_indices, val_indices]

        for beta in candidate_betas, zeta in candidate_zetas

            sol = alternating_projection(train_data, k_rank; beta_mult=beta,
                                         zeta_mult=zeta)

            val_estimate = LL_block_data * pinv(sol[1]) * UR_block_data
            val_error = norm(val_estimate - val_data)^2/norm(val_data)^2

            param_scores[(beta, zeta)] += val_error / num_samples

        end
    end

    best_score = 1e9
    best_params = ()
    for (param, score) in param_scores
        if score < best_score
            best_score = score
            best_params = param
        end
    end

    return best_params[1], best_params[2], param_scores

end
;
