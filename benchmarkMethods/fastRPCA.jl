include("ScaledGD.jl")

function f_RPCA_projection(A, radius)

    n, m = size(A)

    projected_matrix = zeros(n, m)
    for i=1:n
        row_norm = norm(A[i, :])
        if row_norm <= radius
            projected_matrix[i, :] = A[i, :]
        else
            projected_matrix[i, :] = A[i, :] / row_norm * radius
        end
    end

    return projected_matrix
end


function fast_RPCA(A, k_rank, k_sparse; gamma=2, max_iteration=1000)
    """
    This function computes a feasible solution to Robust PCA by employing the
    fast RPCA algorithm as described in "Fast Algorithms for Robust PCA via
    Gradient Descent" (Yi et al. 2016).

    :param A: An arbitrary n-by-n matrix.
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param gamma: A hyperparameter that controls the sparse matrix thresholding
                  function (Float64).
    :param max_iteration: The number of iterations of the main optimization
                          loop to execute (Int64).

    :return: This funciton returns a tuple of two n-by-n arrays that correspond
             to the feasible solution found by this method (the first element
             in the tuple is the matrix X and the second element is the
             matrix Y).
    """
    n = size(A)[1]
    alpha = k_sparse / n^2

    U, sigma, Vt = svd(A)

    # Compute "empirical" incoherence parameter
    incoherence_param = 0
    for i=1:n

        incoherence_param = maximum([norm(U[i, :]) * sqrt(n / k_rank),
                                     norm(Vt[:, i]) * sqrt(n / k_rank),
                                     incoherence_param])

    end

    L, S, R = tsvd(A, k_rank)

    U_naught = L * Diagonal(sqrt.(S))
    V_naught = R * Diagonal(sqrt.(S))

    _, U_s, _ = svd(U_naught)
    _, V_s, _ = svd(V_naught)

    U_radius = sqrt(2 * incoherence_param * k_rank / n) * U_s[1]
    V_radius = sqrt(2 * incoherence_param * k_rank / n) * V_s[1]

    # Initilize the sparse matrix, U and V iterates
    Y_iterate = sparse_threshold(A, alpha)
    U_iterate = f_RPCA_projection(U_naught, U_radius)
    V_iterate = f_RPCA_projection(V_naught, V_radius)

    eta = 1 / tsvd(U_naught * V_naught')[2][1]

    # Main loop
    for iteration=1:max_iteration
        # Update the sparse matrix iterate
        Y_iterate = sparse_threshold(A - U_iterate * V_iterate', gamma * alpha)

        gradients = RPCA_gradient(U_iterate, V_iterate, Y_iterate, A)
        residual = U_iterate' * U_iterate - V_iterate' * V_iterate
        U_update = U_iterate - eta * (gradients[1] + U_iterate * residual / 2)
        V_update = V_iterate - eta * (gradients[2] - V_iterate * residual / 2)

        # Update the U and V iterates
        U_iterate = f_RPCA_projection(U_update, U_radius)
        V_iterate = f_RPCA_projection(V_update, V_radius)
    end

    return (U_iterate * V_iterate', Y_iterate)
end

function cross_validate_fRPCA(U, k_sparse, k_rank;
                              num_samples=30, train_frac=0.7,
                              candidate_gammas=[10, 8, 6, 4, 2, 1,
                                                0.5, 0.1, 0.05, 0.01])
    """
    This function performs cross validation to select the regularization
    parameters lambda and mu for the optimization problem given by
        min ||U - X - Y||_F^2 + lambda * ||X||_F^2 + mu * ||Y||_F^2
        subject to rank(X) <= k_rank, ||Y||_0 <= k_sparse

    :param U: An arbitrary n-by-n matrix.
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
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
    for gamma in candidate_gammas
        param_scores[gamma] = 0
    end
    bad_keys = []

    for trial=1:num_samples

        permutation = randperm(n)
        val_indices = permutation[1:val_dim]
        train_indices = permutation[(val_dim+1):end]

        val_data = U[val_indices, val_indices]
        train_data = U[train_indices, train_indices]

        LL_block_data = U[val_indices, train_indices]
        UR_block_data = U[train_indices, val_indices]

        for gamma in candidate_gammas

            sol = fast_RPCA(train_data, k_rank, k_sparse, gamma=gamma)

            psuedo_inv = nothing
            try
                psuedo_inv = pinv(sol[1])
            catch
              println("pinv threw an error.")
            end

            if psuedo_inv == nothing
                append!(bad_keys, gamma)
            end

            val_estimate = LL_block_data * pseudo_inv * UR_block_data
            val_error = norm(val_estimate - val_data)^2/norm(val_data)^2

            param_scores[gamma] += val_error / num_samples

        end
    end

    best_score = 1e9
    best_param = nothing
    for (param, score) in param_scores
        if param in bad_keys
            continue
        end
        if score < best_score
            best_score = score
            best_param = param
        end
    end

    return best_param, param_scores

end
;
