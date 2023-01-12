function sparse_threshold(A, alpha)
    """
    This function perfoms thresholding of the matrix A as described in
    "Accelerating Ill-Conditioned Low-Rank Matrix Estimation via Scaled
    Gradient Descent" (Tong et al. 2021) with respect to the float alpha.

    :param A: An arbitrary n-by-n matrix.
    :param alpha: The desired thresholding value (Float64).

    :return: The n-by-n thresholding matrix of A.
    """

    n = size(A)[1]
    index = Int(floor(alpha * n))
    row_maximums = zeros(n)
    col_maximums = zeros(n)

    # Identify the index^{th} largest element in each row and column
    for i=1:n
        row_maximums[i] = sort(vec(broadcast(abs, A[i, :])), rev=true)[index]
        col_maximums[i] = sort(vec(broadcast(abs, A[:, i])), rev=true)[index]
    end

    # Execute the thresholding
    row_filter = zeros(Bool, (n, n))
    col_filter = zeros(Bool, (n, n))
    for i=1:n
        row_filter[i, :] = abs.(A[i, :]) .>= row_maximums[i]
        col_filter[:, i] = abs.(A[:, i]) .>= col_maximums[i]
    end
    filter = row_filter .& col_filter

    return A .* filter

end

function RPCA_gradient(U, V, S, A)
    """
    This function evaluates the partial gradients of the RPCA objective
    function when the low-rank matrix is factorized as X = U * V^T.

    :param U: An n-by-n matrix corresponding to the U factor of X.
    :param V: An n-by-n matrix corresponding to the V factor of X.
    :param S: An n-by-n matrix corresponding to the sparse matrix.
    :param A: An arbitrary n-by-n matrix (the input data matrix to RPCA).

    :return: A 1-dimensional vector of length 2. The first component of the
             vector is the partial with respect to U. The second component is
             the partial with respect to V.
    """

    X_iterate = U * V'
    U_grad = (S + X_iterate - A) * V
    V_grad = (S' + X_iterate' - A') * U

    return (U_grad, V_grad)

end
;

function scaled_GD(A, k_rank, k_sparse; gamma=2, max_iteration=1000)
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

        gradients = RPCA_gradient(U_iterate, V_iterate, Y_iterate, A)
        U_update = U_iterate - eta * gradients[1] * pinv(V_iterate' * V_iterate)
        V_update = V_iterate - eta * gradients[2] * pinv(U_iterate' * U_iterate)

        # Update the U and V iterates
        U_iterate = U_update
        V_iterate = V_update
    end

    return (U_iterate * V_iterate', Y_iterate)
end
