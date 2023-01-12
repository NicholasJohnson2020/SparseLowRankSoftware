function sparse_threshold(A, alpha)

    n = size(A)[1]
    index = Int(floor(alpha * n))
    row_maximums = zeros(n)
    col_maximums = zeros(n)

    for i=1:n
        row_maximums[i] = sort(vec(broadcast(abs, A[i, :])), rev=true)[index]
        col_maximums[i] = sort(vec(broadcast(abs, A[:, i])), rev=true)[index]
    end

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

    X_iterate = U * V'
    U_grad = (S + X_iterate - A) * V
    V_grad = (S' + X_iterate' - A') * U

    return (U_grad, V_grad)

end
