function stable_principal_component_pursuit(M, sigma;
    threshold = 1e-6, solver_output=0, use_mosek=false)

    n = size(M)[1]
    mu = (2*n)^0.5 * sigma

    if use_mosek
        m = Model(Mosek.Optimizer)
        set_optimizer_attribute(m, "MSK_IPAR_LOG", solver_output)
    else
        m = Model(SCS.Optimizer)
        set_optimizer_attribute(m, "verbose", solver_output)
    end

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
