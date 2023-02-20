include("generateSyntheticData.jl")

function compute_objective_value(X, Y, U, mu, lambda)
    """
    This function computes the objective value of the optimization problem
    for the solution given by (X, Y). The objective value is given by:
        ||U - X - Y||_F^2 + lambda * ||X||_F^2 + mu * ||Y||_F^2

    :param X: An n-by-n matrix.
    :param Y: An n-by-n matrix.
    :param U: An n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
               (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).

    :return: The objective value (Float64).
    """
    val = norm(U-X-Y)^2 + lambda * norm(X)^2 + mu * norm(Y)^2

    return val

end;

function is_feasible(X, Y, k_sparse, k_rank; epsilon = 1e-10)
    """
    This function verifies that the solution (X, Y) is feasible under the
    constraints rank(X) <= k_rank and ||Y||_0 <= k_sparse. Returns true if
    these constraints are satisfied, false otherwise.

    :param X: An n-by-n matrix.
    :param Y: An n-by-n matrix.
    :paeam k_sparse: The maximum number of non-zeros elements in Y (Int64).
    :param k_rank: The maximum rank of the X (Int64).

    :return: True if the constraints are satisfied, false otherwise (Bool).
    """
    # Verify the sparsity constraint
    num_nonzero = count(abs.(Y) .> epsilon)
    if num_nonzero > k_sparse
        return false
    end

    # Verify the rank constraint
    rank_X = rank(X)
    if rank_X > k_rank
        return false
    end

    return true

end;

function project_matrix(A, k_rank; exact_svd=true)
    """
    This function projects the matrix A onto its first k_rank principal
    components.

    :param A: An arbitrary n-by-n matrix.
    :param k_rank: The desired rank of the projection (Int64).
    :param exact_svd: If true, compute an exact truncated SVD. If false,
                      approximate the SVD using a randomized algorithm (Bool).

    :return: The n-by-n low rank projection of A.
    """
    if exact_svd

        factors = tsvd(A, k_rank)
        U = factors[1]
        S = factors[2]
        V = factors[3]

        projected_matrix = U * Diagonal(S) * V'

    else

        projected_matrix = pqrfact(A, rank=k_rank)
        projected_matrix = Matrix(projected_matrix)
        S = nothing

    end

    return projected_matrix

end;

function construct_binary_matrix(A, k_sparse; zero_indices=[], one_indices=[])
    """
    This function constructs a binary matrix S of the same shape as A where
    S_ij = 1 if and only if A_ij is one of the k_sparse largest entries of A in
    absolute value.

    :param A: An arbitrary n-by-n matrix.
    :param k_sparse: The desired sparsity of the output matrix (Int64).
    :param zero_indices: List of 2-tuples of Int64 where each entry consists of
                         an index (i, j) such that S_ij is constrained to take
                         value 0.
    :param one_indices: List of 2-tuples of Int64 where each entry consists of
                        an index (i, j) such that S_ij is constrained to take
                        value 1.

    :return: An n-by-n binary matrix.
    """
    n = size(A)[1]

    num_zeros = size(zero_indices)[1]
    num_ones = size(one_indices)[1]

    # Verify that zero_indices and one_indices have a valid length
    @assert num_ones <= k_sparse
    @assert num_zeros <= n * n - k_sparse

    reverse_sorted_entries = sortperm(vec(broadcast(abs, A)), rev=true)

    S = zeros(n, n)
    for (i, j) in one_indices
        S[i, j] = 1
    end

    num_selected_indices = num_ones
    reverse_sorted_index = 1

    while num_selected_indices < k_sparse
        index = reverse_sorted_entries[reverse_sorted_index]
        i = Int64((index - 1) % n + 1)
        j = Int64(floor((index - 1) / n)) + 1

        if S[i, j] == 0 && !((i, j) in zero_indices)
            S[i, j] = 1
            num_selected_indices = num_selected_indices + 1
        end
        reverse_sorted_index = reverse_sorted_index + 1
    end

    return S

end;

function solve_sparse_problem(U_tilde, mu, k_sparse;
    zero_indices=[], one_indices=[])
    """
    This function exactly solves the problem of approximating (in squared
    frobenius norm) an arbitrary matrix U_tilde by a sparse matrix and under a
    regularization penalty.

    :param U_tilde: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
               (Float64).
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param zero_indices: List of 2-tuples of Int64 where each entry consists of
                         an index (i, j) such that Y_ij is constrained to take
                         value 0.
    :param one_indices: List of 2-tuples of Int64 where each entry consists of
                        an index (i, j) such that S_ij is constrained to take
                        value 1 where S is the binary matrix denoting the
                        sparsity pattern of Y.

    :return: The n-by-n sparse matrix that solves the problem.
    """
    S = construct_binary_matrix(U_tilde, k_sparse,
        zero_indices = zero_indices, one_indices = one_indices)

    Y = (S .* U_tilde) ./ (1 + mu)

    return Y

end;

function solve_rank_problem(U_tilde, lambda, k_rank; exact_svd=true)
    """
    This function solves the problem of approximating (in squared
    frobenius norm) an arbitrary matrix U_tilde by a low rank matrix and under
    a regularization penalty.

    :param U_tilde: An arbitrary n-by-n matrix.
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param exact_svd: If true, compute an exact truncated SVD. If false,
                      approximate the SVD using a randomized algorithm (Bool).

    :return: The n-by-n low rank matrix that solves the problem.
    """
    return project_matrix(U_tilde ./ (1 + lambda), k_rank, exact_svd=exact_svd)

end;

function iterate_X_Y(U, mu, lambda, k_sparse, k_rank, X_init, Y_init;
    zero_indices = [], one_indices = [], min_improvement=0.001, exact_svd=true)
    """
    This function computes a feasible solution to the problem
        min ||U - X - Y||_F^2 + lambda * ||X||_F^2 + mu * ||Y||_F^2
        subject to rank(X) <= k_rank, ||Y||_0 <= k_sparse

    by iteratively solving the sparse subproblem and the rank subproblem
    starting from an initial feasible solution (X_init, Y_init).

    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
                   (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param X_init: An n-by-n matrix with rank at most k_rank.
    :param Y_init: An n-by-n matrix with sparsity at most k_sparse.
    :param zero_indices: List of 2-tuples of Int64 where each entry consists of
                         an index (i, j) such that Y_ij is constrained to take
                         value 0.
    :param one_indices: List of 2-tuples of Int64 where each entry consists of
                        an index (i, j) such that S_ij is constrained to take
                        value 1 where S is the binary matrix denoting the
                        sparsity pattern of Y.
    :param min_improvement: The minimal fractional decrease in the objective
                            value required for the procedure to continue
                            iterating.
    :param exact_svd: If true, compute exact SVDs. If false, compute
                      approximate SVDs (Bool).

    :return: This function returns three values. The first value is a tuple of
             two n-by-n arrays that correspond to the feasible solution found
             by this method (the first element in the tuple is the matrix X and
             the second element is the matrix Y). The second value is the
             objective value (Float64) achieved by the returned feasible
             solution. The third value is the number of iterations performed by
             this method (Int64).
    """
    old_objective = compute_objective_value(X_init, Y_init, U, mu, lambda)

    Y = solve_sparse_problem(U - X_init, mu, k_sparse,
                             zero_indices=zero_indices, one_indices=one_indices)
    X = solve_rank_problem(U - Y, lambda, k_rank, exact_svd=exact_svd)

    new_objective = compute_objective_value(X, Y, U, mu, lambda)
    step_count = 1

    # If the last step improved the solution more than min_provement,
    # take another step
    while((old_objective - new_objective) / old_objective > min_improvement)

        Y = solve_sparse_problem(U - X, mu, k_sparse,
                        zero_indices=zero_indices, one_indices=one_indices)
        X = solve_rank_problem(U - Y, lambda, k_rank, exact_svd=exact_svd)

        old_objective = new_objective
        new_objective = compute_objective_value(X, Y, U, mu, lambda)
        step_count = step_count + 1

    end

    return (X, Y), new_objective, step_count

end;

# TODO: edit function description to describe zero_indices and one_indices params
function SLR_AM(U, mu, lambda, k_sparse, k_rank; zero_indices=[],
    one_indices=[], random_restarts=100, min_improvement=0.001, exact_svd=true)
    """
    This function computes a feasible solution to the problem
        min ||U - X - Y||_F^2 + lambda * ||X||_F^2 + mu * ||Y||_F^2
        subject to rank(X) <= k_rank, ||Y||_0 <= k_sparse

    by randomly initializing a feasible solution (X, Y) and iteratively solving
    the rank subproblem and the sparsity subproblem until convergence. This is
    repeated random_restarts times and the best solution is returned.

    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
                   (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param zero_indices: List of 2-tuples of Int64 where each entry consists of
                         an index (i, j) such that Y_ij is constrained to take
                         value 0.
    :param one_indices: List of 2-tuples of Int64 where each entry consists of
                        an index (i, j) such that S_ij is constrained to take
                        value 1 where S is the binary matrix denoting the
                        sparsity pattern of Y.
    :param random_restarts: The number of times to repeat the optimizization
                            process from a random initial feasible solution
                            (Int64).
    :param min_improvement: The minimal fractional decrease in the objective
                            value required for the minimization procedure to
                            continue iterating.
    :param exact_svd: If true, compute exact SVDs. If false, compute
                      approximate SVDs (Bool).

    :return: This function returns two values. The first value is a tuple of two
             n-by-n arrays that correspond to the best feasible solution found
             by this method (the first element in the tuple is the matrix X and
             the second element is the matrix Y). The second value is the
             objective value (Float64) achieved by the returned feasible
             solution.
    """
    # Initialize X and Y
    n = size(U)[1]

    X_init = zeros(n, n)
    Y_init = zeros(n, n)

    best_sol, best_obj, _ = iterate_X_Y(U, mu, lambda, k_sparse, k_rank,
                                        X_init, Y_init,
                                        zero_indices=zero_indices,
                                        one_indices=one_indices,
                                        min_improvement=min_improvement,
                                        exact_svd=exact_svd)

    # Repeat the search process random_restarts times
    for i=1:random_restarts
        X_random = rand(Float64, (n, n))
        X_init = project_matrix(X_random, k_rank)
        Y_random = rand(Float64, (n, n))
        Y_init = construct_binary_matrix(Y_random, k_sparse) .* Y_random

        new_sol, new_obj, _ = iterate_X_Y(U, mu, lambda, k_sparse, k_rank,
                                          X_init, Y_init,
                                          zero_indices=zero_indices,
                                          one_indices=one_indices,
                                          min_improvement=min_improvement,
                                          exact_svd=exact_svd)

        # Store the newly found solution if it is better than the best found
        # solution to this point
        if new_obj < best_obj
            best_obj = new_obj
            best_sol = new_sol
        end
    end

    # Verify that the solution to be returned is in fact feasible
    @assert is_feasible(best_sol[1], best_sol[2], k_sparse, k_rank)

    return best_sol, best_obj

end;

function cross_validate_AM(U, k_sparse, k_rank;
                           num_samples=30, train_frac=0.7,
                           candidate_lambdas=[10, 1, 0.1, 0.01],
                           candidate_mus=[10, 1, 0.1, 0.01],
                           exact_svd=true)
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
    :param candidate_lambdas: List of values of lambda (Float64) to be
                              considered during cross validation.
    :param candidate_mus: List of values of mu (Float64) to be considered during
                          cross validation.
    :param exact_svd: If true, compute exact SVDs. If false, compute
                      approximate SVDs (Bool).

    :return: This function returns 3 values.
             1) The best performing lambda value (Float64).
             2) The best performing mu value (Float64).
             3) A dictionary of cross validation scores of all parameters.
    """

    n = size(U)[1]
    val_dim = Int(floor(n * (1-sqrt(train_frac))))
    train_dim = n - val_dim

    param_scores = Dict()
    for lambda_mult in candidate_lambdas, mu_mult in candidate_mus
        param_scores[(lambda_mult, mu_mult)] = 0
    end

    for trial=1:num_samples

        permutation = randperm(n)
        val_indices = permutation[1:val_dim]
        train_indices = permutation[(val_dim+1):end]

        val_data = U[val_indices, val_indices]
        train_data = U[train_indices, train_indices]

        LL_block_data = U[val_indices, train_indices]
        UR_block_data = U[train_indices, val_indices]

        for lambda_mult in candidate_lambdas, mu_mult in candidate_mus

            this_lambda = lambda_mult / n^0.5
            this_mu = mu_mult / n^0.5

            sol, _ = SLR_AM(train_data, this_mu, this_lambda, k_sparse, k_rank,
                            exact_svd=exact_svd)

            val_estimate = LL_block_data * pinv(sol[1]) * UR_block_data
            val_error = norm(val_estimate - val_data)^2/norm(val_data)^2

            param_scores[(lambda_mult, mu_mult)] += val_error / num_samples

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

end;


function validate_AM_params(n, param_frac, sigma; signal_to_noise=10,
                            exact_svd=true, train_size=20,
                            candidate_lambdas=[10, 1, 0.1, 0.01],
                            candidate_mus=[10, 1, 0.1, 0.01],
                            fixed_rank_sparse=false,
                            fixed_rank=1, fixed_sparse=0,
                            symmetric_flag=false)
    """
    This function performs cross validation to select the regularization
    parameters lambda and mu for the optimization problem given by
        min ||U - X - Y||_F^2 + lambda * ||X||_F^2 + mu * ||Y||_F^2
        subject to rank(X) <= k_rank, ||Y||_0 <= k_sparse

    :param n: The row and column dimension of the data (Int64).
    :param param_frac: Parameter that controls the rank and sparsity of the
                       sampled low rank and sparse matrices when
                       fixed_rank_sparse=false (Float64).
    :param sigma: Parameter that controls the absolute magnitude of the noise
                  (Float64).
    :param signal_to_noise: Parameter that controls the signal to noise ratio of
                            the sampled data (Float64).
    :param exact_svd: If true, compute exact SVDs. If false, compute
                      approximate SVDs (Bool).
    :param train_size: The number of samples to draw per parameter when
                       performing cross validation (Int64).
    :param candidate_lambdas: List of values of lambda (Float64) to be
                              considered during cross validation.
    :param candidate_mus: List of values of mu (Float64) to be considered during
                          cross validation.
    :param fixed_rank_sparse: If true, use fixed_rank and fixed_sparse as the
                              rank and sparsity respectively of the low rank
                              matrix and the sparse matrix when sampling (Bool).
    :param fixed_rank: Rank of the low rank matrices to be sampled when
                       fixed_rank_sparse=true.
    :param fixed_sparse: Sparsity of the sparse matrices to be sampled when
                         fixed_rank_sparse=true
    :param symmetric_flag: If true, sampling symmetric data (Bool).

    :return: This function returns 2 values.
             1) The best performing lambda value (Float64).
             2) The best performing mu value (Float64).
    """

    param_scores = Dict()
    for lambda_mult in candidate_lambdas, mu_mult in candidate_mus

        param_scores[(lambda_mult, mu_mult)] = 0
        this_lambda = lambda_mult / n^0.5
        this_mu = mu_mult / n^0.5

        for trial = 1:train_size

            if symmetric_flag

                D, L_0, S_0, k_sparse, k_rank = generate_synthetic_data_symmetric(n,
                                                    param_frac, sigma,
                                                    signal_to_noise=signal_to_noise,
                                                    fixed_rank_sparse=fixed_rank_sparse,
                                                    fixed_rank=fixed_rank,
                                                    fixed_sparse=fixed_sparse)

            else

                D, L_0, S_0, k_sparse, k_rank = generate_synthetic_data(n,
                                                    param_frac, sigma,
                                                    signal_to_noise=signal_to_noise,
                                                    fixed_rank_sparse=fixed_rank_sparse,
                                                    fixed_rank=fixed_rank,
                                                    fixed_sparse=fixed_sparse)
            end

            sol, _ = SLR_AM(D, this_mu, this_lambda, k_sparse, k_rank,
                            exact_svd=exact_svd)

            D_error = norm(D - sol[1] - sol[2])^2 / norm(D)^2
            L_error = norm(L_0 - sol[1])^2 / norm(L_0)^2
            S_error = norm(S_0 - sol[2])^2 / norm(S_0)^2
            score = D_error + L_error + S_error

            param_scores[(lambda_mult, mu_mult)] += score

        end

        param_scores[(lambda_mult, mu_mult)] /= train_size

    end

    best_score = 1000
    best_params = ()
    for (param, score) in param_scores
        if score < best_score
            best_score = score
            best_params = param
        end
    end

    return best_params[1], best_params[2]

end;
