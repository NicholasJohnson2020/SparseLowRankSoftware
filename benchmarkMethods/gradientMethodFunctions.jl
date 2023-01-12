function sparse_threshold(A, alpha)
    """
    This function perfoms thresholding of the matrix A as described in "Fast
    Algorithms for Robust PCA via Gradient Descent" (Yi et al. 2016) with
    respect to the float alpha.

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
