function stable_principal_component_pursuit(M, sigma;
    threshold = 1e-6, solver_output=0)
    """
    This function solves the stable principal component pursuit problem.

    :param M: An arbitrary n-by-n matrix.
    :param sigma: A parameter that controls the tradeoff between nuclear norm
                  term and the L1 norm term in the objective function
    :param threshold: Singular values in the output low rank matrix below
                      threshold are set to 0. Entries in the output sparse
                      matrix below threshold are set to 0 (Float64).
    :param solver_output: Solver_output param to be passed to SCS (Int64).

    :return: This function returns two values. The first value is a tuple of 2
             n-by-n arrays that correspond to the solution of stable PCP (the
             first element in the tuple is the matrix X and the second element
             is the matrix Y). The second value is the optimal objective value
             of the optimization problem (Float64).
    """
    n = size(M)[1]
    mu = (2 * n) ^ 0.5 * sigma

    # Build stable PCP formulation using JuMP
    m = Model(SCS.Optimizer)
    set_optimizer_attribute(m, "verbose", solver_output)

    @variable(m, L[i=1:n, j=1:n])
    @variable(m, W_1[i=1:n, j=1:n])
    @variable(m, W_2[i=1:n, j=1:n])
    @variable(m, S[i=1:n, j=1:n])
    @variable(m, S_abs[i=1:n, j=1:n])
    @variable(m, error[i=1:n, j=1:n])

    @constraint(m, [W_1 L; L' W_2] in PSDCone())
    @constraint(m, [i=1:n, j=1:n], S_abs[i, j] >= S[i, j])
    @constraint(m, [i=1:n, j=1:n], S_abs[i, j] >= -S[i, j])

    @constraint(m, [i=1:n, j=1:n], (M[i, j] - L[i, j] - S[i, j])^2 <= error[i, j])

    @objective(m, Min, 0.5 * sum(W_1[i, i] for i=1:n) +
        0.5 * sum(W_2[i, i] for i=1:n) +
        n^(-0.5) * sum(S_abs[i, j] for i=1:n, j=1:n) +
        sum(error[i, j] for i=1:n, j=1:n)/(2*mu))

    # Solve stable PCP
    optimize!(m)

    L_opt = value.(L)
    S_opt = value.(S)

    L_svd = svd(L_opt)
    L_U = L_svd.U
    L_S = L_svd.S
    L_Vt = L_svd.Vt

    for i = 1:n
        if abs(L_S[i]) < threshold
            L_S[i] = 0
        end
    end

    L_opt = L_U * Diagonal(L_S) * L_Vt

    for i = 1:n
        for j = 1:n
            if abs(S_opt[i, j]) < threshold
                S_opt[i, j] = 0
            end
        end
    end

    return (L_opt, S_opt), objective_value(m)

end;
