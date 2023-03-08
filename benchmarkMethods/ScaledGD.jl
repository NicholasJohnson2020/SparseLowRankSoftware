function sparse_threshold(A, alpha)
    """
    This function perfoms thresholding of the matrix A as described in
    "Accelerating Ill-Conditioned Low-Rank Matrix Estimation via Scaled
    Gradient Descent" (Tong et al. 2021) with respect to the float alpha.

    :param A: An arbitrary n-by-n matrix.
    :param alpha: The desired thresholding value (Float64).

    :return: The n-by-n thresholding matrix of A.
    """

    n = size(A)[1]
    index = maximum([Int(floor(alpha * n)), 1])
    index = minimum([index, n])
    row_maximums = zeros(n)
    col_maximums = zeros(n)

    # Identify the index^{th} largest element in each row and column
    for i=1:n
        row_maximums[i] = sort(vec(broadcast(abs, A[i, :])), rev=true)[index]
        col_maximums[i] = sort(vec(broadcast(abs, A[:, i])), rev=true)[index]
    end

    # Execute the thresholding
    row_filter = zeros(Bool, (n, n))
    col_filter = zeros(Bool, (n, n))
    for i=1:n
        row_filter[i, :] = abs.(A[i, :]) .>= row_maximums[i]
        col_filter[:, i] = abs.(A[:, i]) .>= col_maximums[i]
    end
    filter = row_filter .& col_filter

    return A .* filter

end

function RPCA_gradient(U, V, S, A)
    """
    This function evaluates the partial gradients of the RPCA objective
    function when the low-rank matrix is factorized as X = U * V^T.

    :param U: An n-by-n matrix corresponding to the U factor of X.
    :param V: An n-by-n matrix corresponding to the V factor of X.
    :param S: An n-by-n matrix corresponding to the sparse matrix.
    :param A: An arbitrary n-by-n matrix (the input data matrix to RPCA).

    :return: A 1-dimensional vector of length 2. The first component of the
             vector is the partial with respect to U. The second component is
             the partial with respect to V.
    """

    X_iterate = U * V'
    U_grad = (S + X_iterate - A) * V
    V_grad = (S' + X_iterate' - A') * U

    return (U_grad, V_grad)

end
;

function scaled_GD(A, k_rank, k_sparse; gamma=2, max_iteration=1000,
                   termination_criteria="rel_improvement",
                   min_improvement=0.001)
    """
    This function computes a feasible solution to Robust PCA by employing the
    ScaledGD algorithm as described in "Accelerating Ill-Conditioned Low-Rank
    Matrix Estimation via Scaled Gradient Descent" (Tong et al. 2021).

    :param A: An arbitrary n-by-n matrix.
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param gamma: A hyperparameter that controls the sparse matrix thresholding
                  function (Float64).
    :param max_iteration: The number of iterations of the main optimization
                          loop to execute (Int64).
    :param termination_criteria: String that must take value either
                                 "rel_improvement" or "iteration_count". If
                                 "rel_improvement", the algorithm will terminate
                                 if the fractional decrease in the objective
                                 value after an iteration is less than
                                 min_improvement. If set to "iteration_count"
                                 the algorithm will terminate after
                                 max_iteration steps (String).
    :param min_improvement: The minimal fractional decrease in the objective
                            value required for the procedure to continue
                            iterating when termination_criteria is set to
                            "rel_improvement".

    :return: This funciton returns a tuple of two n-by-n arrays that correspond
             to the feasible solution found by this method (the first element
             in the tuple is the matrix X and the second element is the
             matrix Y).
    """

    @assert termination_criteria in ["iteration_count", "rel_improvement"]

    n = size(A)[1]
    alpha = k_sparse / n^2

    Y_iterate = sparse_threshold(A, gamma * alpha)

    L, S, R = tsvd(A, k_rank)

    U_iterate = L * Diagonal(sqrt.(S))
    V_iterate = R * Diagonal(sqrt.(S))

    old_objective = 0
    new_objective = 0
    if termination_criteria == "rel_improvement"
        new_objective = compute_objective_value(U_iterate * V_iterate',
                                                Y_iterate, A, 0, 0)
    end

    # Default step size specified in the paper
    eta = 2 / 3

    # Main loop
    for iteration=1:max_iteration
        # Update the sparse matrix iterate
        Y_iterate = sparse_threshold(A - U_iterate * V_iterate', gamma * alpha)

        gradients = RPCA_gradient(U_iterate, V_iterate, Y_iterate, A)
        U_update = U_iterate - eta * gradients[1] * pinv(V_iterate' * V_iterate)
        V_update = V_iterate - eta * gradients[2] * pinv(U_iterate' * U_iterate)

        # Update the U and V iterates
        U_iterate = U_update
        V_iterate = V_update

        if termination_criteria == "rel_improvement"
            old_objective = new_objective
            new_objective = compute_objective_value(U_iterate * V_iterate',
                                                    Y_iterate, A, 0, 0)
            if (old_objective - new_objective) / old_objective < min_improvement
                break
            end
        end
    end

    return (U_iterate * V_iterate', Y_iterate)
end;

function cross_validate_ScaledGD(U, k_sparse, k_rank;
                                 num_samples=30, train_frac=0.7,
                                 candidate_gammas=[10, 8, 6, 4, 2, 1,
                                                   0.5, 0.1, 0.05, 0.01],
                                 termination_criteria="rel_improvement",
                                 max_iteration=1000,
                                 min_improvement=0.001)
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
    :param max_iteration: The number of iterations of the main optimization
                          loop to execute (Int64).
    :param termination_criteria: String that must take value either
                                 "rel_improvement" or "iteration_count". If
                                 "rel_improvement", the algorithm will terminate
                                 if the fractional decrease in the objective
                                 value after an iteration is less than
                                 min_improvement. If set to "iteration_count"
                                 the algorithm will terminate after
                                 max_iteration steps (String).
    :param min_improvement: The minimal fractional decrease in the objective
                            value required for the procedure to continue
                            iterating when termination_criteria is set to
                            "rel_improvement".

    :return: This function returns 2 values.
             1) The best performing gamma value (Float64).
             2) A dictionary of cross validation scores of all parameters.
    """

    @assert termination_criteria in ["iteration_count", "rel_improvement"]

    n = size(U)[1]
    val_dim = Int(floor(n * (1-sqrt(train_frac))))
    train_dim = n - val_dim

    param_scores = Dict()
    for gamma in candidate_gammas
        param_scores[gamma] = 0
    end

    for trial=1:num_samples

        permutation = randperm(n)
        val_indices = permutation[1:val_dim]
        train_indices = permutation[(val_dim+1):end]

        val_data = U[val_indices, val_indices]
        train_data = U[train_indices, train_indices]

        LL_block_data = U[val_indices, train_indices]
        UR_block_data = U[train_indices, val_indices]

        for gamma in candidate_gammas

            sol = scaled_GD(train_data, k_rank, k_sparse, gamma=gamma,
                            termination_criteria=termination_criteria,
                            max_iteration=max_iteration,
                            min_improvement=min_improvement)

            val_estimate = LL_block_data * pinv(sol[1]) * UR_block_data
            val_error = norm(val_estimate - val_data)^2/norm(val_data)^2

            param_scores[gamma] += val_error / num_samples

        end
    end

    best_score = 1e9
    best_param = nothing
    for (param, score) in param_scores
        if score < best_score
            best_score = score
            best_param = param
        end
    end

    return best_param, param_scores

end;
