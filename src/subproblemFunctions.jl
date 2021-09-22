using Pkg
#Pkg.activate("/home/nagj/.julia/environments/sparse_discrete")
include("convexRelaxations.jl")

# TODO: edit function description to describe zero_indices and one_indices params
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

    :return: The n-by-n sparse matrix that solves the problem.
    """
    S = construct_binary_matrix(U_tilde, k_sparse,
        zero_indices = zero_indices, one_indices = one_indices)

    Y = (S .* U_tilde) ./ (1 + mu)

    return Y

end;

function solve_rank_problem(U_tilde, lambda, k_rank; exact_svd=true)
    """
    This function exactly solves the problem of approximating (in squared
    frobenius norm) an arbitrary matrix U_tilde by a low rank matrix and under
    a regularization penalty.

    :param U_tilde: An arbitrary n-by-n matrix.
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).

    :return: The n-by-n low rank matrix that solves the problem.
    """
    return project_matrix(U_tilde ./ (1 + lambda), k_rank, exact_svd=exact_svd)

end;


function solve_rank_relaxed_problem(U, mu, lambda, k_sparse, k_rank;
    check_feasibility=true)
    """
    This function exactly solves the problem of interest in the absence of the
    low rank constraint. It then rounds this lower bound to a feasible solution.

    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
                   (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).

    :return: This function returns two values. The first value is a tuple of two
             n-by-n arrays that correspond to the solution of the rank
             constraint relaxed problem (the first element in the tuple is the
             matrix X and the second element is the matrix Y). The second value
             is a tuple of two n-by-n arrays that correspond to the rounded
             feasible solution to the original problem (the first element in the
             tuple is the matrix X and the second element is the matrix Y).
    """
    S = construct_binary_matrix(U, k_sparse)

    Y = S .* U
    Y = lambda * Y / (lambda + mu + mu * lambda)

    X = (U - Y) / (1 + lambda)

    # Round to a feasible solution
    projected_X = project_matrix(X, k_rank)

    if check_feasibility
        @assert is_feasible(projected_X, Y, k_sparse, k_rank)
    end

    return (X, Y), (projected_X, Y)

end;

function solve_sparse_relaxed_problem(U, mu, lambda, k_sparse, k_rank;
    check_feasibility=true)
    """
    This function exactly solves the problem of interest in the absence of the
    sparsity constraint. It then rounds this lower bound to a feasible solution.

    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
                   (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).

    :return: This function returns two values. The first value is a tuple of two
             n-by-n arrays that correspond to the solution of the sparsity
             constraint relaxed problem (the first element in the tuple is the
             matrix X and the second element is the matrix Y). The second value
             is a tuple of two n-by-n arrays that correspond to the rounded
             feasible solution to the original problem (the first element in the
             tuple is the matrix X and the second element is the matrix Y).
    """
    projected_U = project_matrix(U, k_rank)

    X = mu * projected_U / (lambda + mu + mu * lambda)

    Y = (U - X) / (1 + mu)

    # Round to a feasible solution
    S = construct_binary_matrix(Y, k_sparse)
    rounded_Y = S .* Y

    if check_feasibility
        @assert is_feasible(X, rounded_Y, k_sparse, k_rank)
    end

    return (X, Y), (X, rounded_Y)

end;

function local_search(X_init, Y_init, U, mu, lambda, k_sparse, k_rank;
    update_sparsity_pattern = true, min_improvement = 0.001)
    """
    This function implements an alternating minimization local search method to
    arrive at a locally optimal solution. Starting from an initial feasible
    solution (X_init, Y_init), this function parametrizes X as X = V * W^T and
    parametrizes Y as Y = S .* (U - V * W^T) / (1 + mu). The function then
    iteratively optimizes over V, W and S until the objective value converges.

    :param X_init: An n-by-n matrix with rank(X_init) <= k_rank.
    Lparam Y_init: An n-by-n matrix with ||Y||_0 <= k_sparse.
    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
               (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param min_improvement: The minimal fractional decrease in the objective
                            value required for the minimization procedure to
                            continue iterating.

    :return: This function returns 4 values. The first return value is a tuple
             of two n-by-n arrays that correspond to the locally optimal
             feasible solution (the first element in the tuple is the matrix X
             and the second element is the matrix Y). The second return value
             is the objective value achieved by the locally optimal solution
             (Float64). The third return value is the number of iterations
             performed (Int64). The final return value is the total fractional
             improvement of the final objective value compared to the initial
             objective value (Float64).
    """
    # Verify that the initial solution (X_init, Y_init) is a feasible solution.
    @assert is_feasible(X_init, Y_init, k_sparse, k_rank)

    # Initialize V, Wt and S
    svd_out = svd(X_init)
    svd_U = svd_out.U
    svd_S = svd_out.S
    svd_Vt = svd_out.Vt

    initial_V = svd_U[:, 1:k_rank] * Diagonal(svd_S[1:k_rank].^0.5)
    initial_Wt = Diagonal(svd_S[1:k_rank].^0.5) * svd_Vt[1:k_rank, :]
    S = abs.(Y_init) .> 1e-10

    old_objective = compute_objective_value(X_init, Y_init, U, mu, lambda)
    init_objective = old_objective

    # Take an initial step in the local search procedure
    V = optimize_V(initial_Wt, S, U, mu, lambda, k_rank)
    Wt = optimize_Wt(V, S, U, mu, lambda, k_rank)
    X_new = V * Wt
    if update_sparsity_pattern
        S = construct_binary_matrix(U - X_new, k_sparse)
    end

    Y_new = S .* (U - X_new)
    Y_new = Y_new / (1 + mu)
    new_objective = compute_objective_value(X_new, Y_new, U, mu, lambda)
    step_count = 1

    # If the last step improved the solution more than min_provement,
    # take another step
    while((old_objective - new_objective) / old_objective > min_improvement)

        V = optimize_V(Wt, S, U, mu, lambda, k_rank)
        Wt = optimize_Wt(V, S, U, mu, lambda, k_rank)
        X_new = V * Wt
        if update_sparsity_pattern
            S = construct_binary_matrix(U - X_new, k_sparse)
        end

        Y_new = S .* (U - X_new)
        Y_new = Y_new / (1 + mu)

        old_objective = new_objective
        new_objective = compute_objective_value(X_new, Y_new, U, mu, lambda)
        step_count = step_count + 1

    end

    total_improvement = (init_objective - new_objective) / init_objective

    return (X_new, Y_new), new_objective, step_count, total_improvement

end;

function optimize_V(Wt, S, U, mu, lambda, k_rank; solver_output=0)
    """
    This function finds the n-by-k_rank matrix V that minimizes the following
    objective:
        ||U - V * Wt -  S .* (U - V * WT) / (1 + mu)||_F^2 +
        lambda * ||V * Wt||_F^2 + mu * ||S .* (U - V * Wt) / (1 + mu)||_F^2

    :param Wt: An arbitrary k_rank-by-n matrix.
    :param S: An arbitrary n-by-n binary matrix.
    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
               (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).

    :return: The optimal n-by-k_rank matrix V.
    """
    n = size(U)[1]

    # Construct the optimization problem in JuMP
    m = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
    set_optimizer_attribute(m, "OutputFlag", solver_output)

    @variable(m, V[i=1:n, j=1:k_rank])
    @variable(m, X[i=1:n, j=1:n])
    @variable(m, Y[i=1:n, j=1:n])

    @constraint(m, X .== V * Wt)
    @constraint(m, [i = 1:n, j = 1:n],
        Y[i, j] == S[i, j] * (U[i, j]-X[i, j]) / (1+mu))

    @objective(m, Min, sum((U[i, j]-X[i, j]-Y[i, j])^2 for i=1:n for j=1:n) +
        lambda * sum(X[i, j]^2 for i=1:n for j=1:n) +
        mu * sum(Y[i, j]^2 for i=1:n for j=1:n))

    optimize!(m)

    return value.(V)

end;

function optimize_Wt(V, S, U, mu, lambda, k_rank; solver_output=0)
    """
    This function finds the K-rank-by-n matrix Wt that minimizes the following
    objective:
        ||U - V * Wt -  S .* (U - V * WT) / (1 + mu)||_F^2 +
        lambda * ||V * Wt||_F^2 + mu * ||S .* (U - V * Wt) / (1 + mu)||_F^2

    :param V: An arbitrary n-by-k_rank matrix.
    :param S: An arbitrary n-by-n binary matrix.
    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
               (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).

    :return: The optimal k_rank-by-n matrix Wt.
    """
    n = size(U)[1]

    # Construct the optimization problem in JuMP
    m = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
    set_optimizer_attribute(m, "OutputFlag", solver_output)

    @variable(m, Wt[i=1:k_rank, j=1:n])
    @variable(m, X[i=1:n, j=1:n])
    @variable(m, Y[i=1:n, j=1:n])

    @constraint(m, X .== V * Wt)
    @constraint(m, [i = 1:n, j = 1:n],
        Y[i, j] == S[i, j] * (U[i, j]-X[i, j]) / (1+mu))

    @objective(m, Min, sum((U[i, j]-X[i, j]-Y[i, j])^2 for i=1:n for j=1:n) +
        lambda * sum(X[i, j]^2 for i=1:n for j=1:n) +
        mu * sum(Y[i, j]^2 for i=1:n for j=1:n))

    optimize!(m)

    return value.(Wt)

end;
