# TODO: edit function description to describe zero_indices and one_indices and
# round_relaxed_sol params and return vals
function solve_perspective_relaxation(U, mu, lambda, k_sparse, k_rank;
    zero_indices=[], one_indices=[], solver_output=0, check_feasibility=true,
    round_relaxed_sol=false, sdp_solver="Mosek")
    """
    This function exactly solves the perspective reformulation relaxation of the
    problem of interest and also rounds this lower bound to a feasible solution.

    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
                   (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).

    :return: This function returns two values. The first value is a tuple of two
             n-by-n arrays that correspond to the solution of the perspective
             relaxation (the first element in the tuple is the matrix X and the
             second element is the matrix Y). The second value is a tuple of
             two n-by-n arrays that correspond to the rounded feasible
             solution to the original problem (the first element in the tuple
             is the matrix X and the second element is the matrix Y).
    """
    n = size(U)[1]

    num_zeros = size(zero_indices)[1]
    num_ones = size(one_indices)[1]

    @assert num_ones <= k_sparse
    @assert num_zeros <= n * n - k_sparse

    # Build perspective formulation relaxation using JuMP
    if sdp_solver == "Mosek"
        m = Model(Mosek.Optimizer)
        set_optimizer_attribute(m, "MSK_IPAR_LOG", solver_output)
    else
        m = Model(SCS.Optimizer)
        set_optimizer_attribute(m, "verbose", solver_output)
    end

    @variable(m, X[i=1:n, j=1:n])
    @variable(m, Y[i=1:n, j=1:n])
    @variable(m, Z[i=1:n, j=1:n] >= 0)
    @variable(m, P[i=1:n, j=1:n])
    @variable(m, P_hat[i=1:n, j=1:n])
    @variable(m, alpha[i=1:n, j=1:n])
    @variable(m, theta[i=1:n, j=1:n] >= 0)

    @constraint(m, [i=1:n, j=1:n], Z[i, j] <= 1)
    @SDconstraint(m, [i=1:n, j=1:n],
        [theta[i, j] Y[i, j]; Y[i, j] Z[i, j]] >= 0)
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

    optimize!(m)

    X_sol = value.(X)
    Y_sol = value.(Y)

    # Round to a feasible solution
    P_sol = value.(P)
    Z_sol = value.(Z)
    alpha_sol = value.(alpha)
    theta_sol = value.(theta)

    if round_relaxed_sol
        rounded_sol, status = round_perspective_relaxation(U, mu, lambda,
                                                k_sparse, k_rank, Z_sol, P_sol)

        if check_feasibility
            @assert is_feasible(rounded_sol[1], rounded_sol[2],
                                k_sparse, k_rank)
        end

        return (X_sol, Y_sol, Z_sol, P_sol), rounded_sol,
               objective_value(m), status

    else

        return (X_sol, Y_sol, Z_sol, P_sol, alpha_sol), objective_value(m)

    end

end;
