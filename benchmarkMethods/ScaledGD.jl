include("gradientMethodFunctions.jl")

function Scaled_GD(A, k_rank, k_sparse; gamma=2, max_iteraton=1000)
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

    :return: This funciton returns a tuple of two n-by-n arrays that correspond
             to the feasible solution found by this method (the first element
             in the tuple is the matrix X and the second element is the
             matrix Y).
    """
    n = size(A)[1]
    alpha = k_sparse / n^2

    Y_iterate = sparse_threshold(A, alpha)

    L, S, R = tsvd(A, k_rank)

    U_iterate = L * Diagonal(sqrt.(S))
    V_iterate = R * Diagonal(sqrt.(S))

    # Default step size specified in the paper
    eta = 2 / 3

    # Main loop
    for iteration=1:max_iteration
        # Update the sparse matrix iterate
        Y_iterate = sparse_threshold(A - U_iterate * V_iterate', gamma * alpha)

        gradients = RPCA_gradient(U_iterate, V_iterate, Y_iterate)
        U_update = U_iterate - eta * gradients[1] * pinv(V_iterate' * V_iterate)
        V_update = V_iterate - eta * gradients[2] * pinv(U_iterate' * U_iterate)

        # Update the U and V iterates
        U_iterate = U_update
        V_iterate = V_update
    end

    return (U_iterate * V_iterate', Y_iterate)
end
