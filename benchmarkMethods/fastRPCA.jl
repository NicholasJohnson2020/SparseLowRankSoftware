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

    U_radius = sqrt(2 * incoherence_param * k_rank / n) * tsvd(U_naught, 1)[2][1]
    V_radius = sqrt(2 * incoherence_param * k_rank / n) * tsvd(V_naught, 1)[2][1]

    # Initilize the sparse matrix, U and V iterates
    Y_iterate = sparse_threshold(A, alpha)
    U_iterate = f_RPCA_projection(U_naught, U_radius)
    V_iterate = f_RPCA_projection(V_naught, V_radius)

    eta = S[1] / 36

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
;
