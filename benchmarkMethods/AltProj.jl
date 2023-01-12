function hard_threshold(A, threshold_param)
    """
    This function perfoms hard threholding of the matrix A with respect to the
    float threshold_param.

    :param A: An arbitrary n-by-n matrix.
    :param threshold_param: The desired hard thresholding value (Float64).

    :return: The n-by-n hard thresholding matrix of A.
    """
    filter_mat = abs.(A) .>= threshold_param
    return A .* filter_mat

end

function alternating_projection(A, k_rank; epsilon=1e-3)
    """
    This function implements the alternating projection algorithm to solve
    sparse plus low rank matrix decomposition.

    :param A: An arbitrary n-by-n matrix.
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param epsilon: A termination parameter for the algorithm (Float64).

    :return: This function returns two values. The first value is a tuple of two
             n-by-n arrays that correspond to the best feasible solution found
             by this method (the first element in the tuple is the matrix X and
             the second element is the matrix Y). The second value is the
             objective value (Float64) achieved by the returned feasible
             solution.
    """
    n = size(A)[1]

    U, sigma, Vt = svd(A)

    # Compute "empirical" incoherence parameter
    incoherence_param = 0
    for i=1:n

        incoherence_param = maximum([norm(U[i, :]) * sqrt(n / k_rank),
                                     norm(Vt[:, i]) * sqrt(n / k_rank),
                                     incoherence_param])

    end

    # Initialize algorithm
    beta = 4 * incoherence_param ^ 2 * k_rank / n
    current_threshold = beta * sigma[1]
    X_iterate = zeros(n, n)
    Y_iterate = hard_threshold(A - X_iterate, current_threshold)

    # Main loop
    for current_rank=1:k_rank

        T = 10 * log(n * beta * norm(A-Y_iterate) / epsilon)

        for iteration=1:(T+1)

            U, sigma, Vt = svd(A - Y_iterate)

            #current_threshold = beta * (sigma[current_rank + 1] +
            #                    0.5 ^ (iteration - 1) * sigma[current_rank])

            # Update sparse matrix threshold
            current_threshold = incoherence_param * sigma[current_rank + 1] / sqrt(n)

            sigma[(current_rank + 1):end] .= 0

            # Perform projections
            X_iterate = U * Diagonal(sigma) * Vt
            Y_iterate = hard_threshold(A - X_iterate, current_threshold)
        end

        if norm(A - X_iterate - Y_iterate) <= epsilon
            return (X_iterate, Y_iterate)
        end
    end

    return (X_iterate, Y_iterate)

end
;
