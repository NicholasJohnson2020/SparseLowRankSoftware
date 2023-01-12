function solve_perspective_relaxation(U, mu, lambda, k_sparse, k_rank;
    zero_indices=[], one_indices=[], solver_output=0)
    """
    This function exactly solves the perspective reformulation relaxation of the
    problem of interest.

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
    :param solver_output: Solver_output param to be passed to Mosek (Int64).

    :return: This function returns two values. The first value is a tuple of 5
             n-by-n arrays that correspond to the solution of the perspective
             relaxation (the first element in the tuple is the matrix X, the
             second element is the matrix Y, the third element is the matrix Z,
             the fourth element is the matrix P and the fifth element is the
             matrix Theta). The second value is the optimal objective value of
             the relaxation (Float64).
    """

    n = size(U)[1]

    num_zeros = size(zero_indices)[1]
    num_ones = size(one_indices)[1]

    @assert num_ones <= k_sparse
    @assert num_zeros <= n * n - k_sparse

    # Build perspective formulation relaxation using JuMP
    m = Model(Mosek.Optimizer)
    set_optimizer_attribute(m, "MSK_IPAR_LOG", solver_output)

    @variable(m, X[i=1:n, j=1:n])
    @variable(m, Y[i=1:n, j=1:n])
    @variable(m, Z[i=1:n, j=1:n] >= 0)
    @variable(m, P[i=1:n, j=1:n])
    @variable(m, P_hat[i=1:n, j=1:n])
    @variable(m, alpha[i=1:n, j=1:n])
    @variable(m, theta[i=1:n, j=1:n] >= 0)

    @constraint(m, [i=1:n, j=1:n], Z[i, j] <= 1)
    @constraint(m, [i=1:n, j=1:n],
        [theta[i, j] Y[i, j]; Y[i, j] Z[i, j]] in PSDCone())
    for (i, j) in zero_indices
        @constraint(m, Z[i, j] == 0)
        @constraint(m, Y[i, j] == 0)
        @constraint(m, theta[i, j] == 0)
    end
    for (i, j) in one_indices
        @constraint(m, Z[i, j] == 1)
        @constraint(m, Y[i, j] == (U[i, j] - X[i, j]) / (1 + mu))
    end
    @constraint(m, sum(Z[i, j] for i=1:n, j=1:n) <= k_sparse)

    @constraint(m, sum(P[i, i] for i = 1:n) <= k_rank)
    @constraint(m, 1*Matrix(I, n, n) - P in PSDCone())
    @constraint(m, [alpha X; X' P] in PSDCone())

    @objective(m, Min, sum((U[i, j]-X[i, j]-Y[i, j])^2 for i=1:n for j=1:n) +
        lambda * sum(alpha[i, i] for i=1:n) +
        mu * sum(theta[i, j] for i=1:n for j=1:n))

    # Solve perspective relaxation
    optimize!(m)

    X_sol = value.(X)
    Y_sol = value.(Y)

    P_sol = value.(P)
    Z_sol = value.(Z)
    alpha_sol = value.(alpha)
    theta_sol = value.(theta)

    return (X_sol, Y_sol, Z_sol, P_sol, alpha_sol), objective_value(m)

end;
