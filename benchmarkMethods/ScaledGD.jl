include("gradientMethodFunctions.jl")

function Scaled_GD(A, k_rank, k_sparse; gamma=2, max_iteraton=1000)

    n = size(A)[1]
    alpha = k_sparse / n^2

    Y_iterate = sparse_threshold(A, alpha)

    L, S, R = tsvd(A, k_rank)

    U_iterate = L * Diagonal(sqrt.(S))
    V_iterate = R * Diagonal(sqrt.(S))

    eta = 2 / 3

    for iteration=1:max_iteration
        Y_iterate = sparse_threshold(A - U_iterate * V_iterate', gamma * alpha)
        gradients = RPCA_gradient(U_iterate, V_iterate, Y_iterate)

        U_update = U_iterate - eta * gradients[1] * pinv(V_iterate' * V_iterate)
        V_update = V_iterate - eta * gradients[2] * pinv(U_iterate' * U_iterate)

        U_iterate = U_update
        V_iterate = V_update
    end

    return (U_iterate * V_iterate', Y_iterate)
end
