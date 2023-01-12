include("gradientMethodFunctions.jl")

function f_RPCA_projection(A, radius)

    n = size(A)[1]

    projected_matrix = zeros(n, n)
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


function fast_RPCA(A, k_rank, k_sparse; gamma=2, max_iteraton=1000)

    n = size(A)[1]
    alpha = k_sparse / n^2

    # Compute "empirical" incoherence parameter
    incoherence_param = 0
    for i=1:n

        incoherence_param = maximum([norm(U[i, :]) * sqrt(n / k_rank),
                                     norm(Vt[:, i]) * sqrt(n / k_rank),
                                     incoherence_param])

    end

    Y_iterate = sparse_threshold(A, alpha)

    L, S, R = tsvd(A, k_rank)

    U_naught = L * Diagonal(sqrt.(S))
    V_naught = R * Diagonal(sqrt.(S))

    U_radius = sqrt(2 * incoherence_param * k_rank / n) * tsvd(U_naught, 1)[2]
    V_radius = sqrt(2 * incoherence_param * k_rank / n) * tsvd(V_naught, 1)[2]

    U_iterate = f_RPCA_projection(U_naught, U_radius)
    V_iterate = f_RPCA_projection(V_naught, V_radius)

    eta = S[1] / 36

    for iteration=1:max_iteration
        Y_iterate = sparse_threshold(A - U_iterate * V_iterate', gamma * alpha)
        gradients = RPCA_gradient(U_iterate, V_iterate, Y_iterate)
        residual = U_iterate' * U_iterate - V_iterate' * V_iterate

        U_update = U_iterate - eta * (gradients[1] + U_iterate * residual / 2)
        V_update = V_iterate - eta * (gradients[2] - V_iterate * residual / 2)

        U_iterate = U_update
        V_iterate = V_update
    end

    return (U_iterate * V_iterate', Y_iterate)
end
;
