include("alternatingMinimization.jl")
include("convexRelaxation.jl")

struct node
    zero_indices::Array{Any, 1}
    one_indices::Array{Any, 1}
    upper_bound::Float64
    lower_bound::Float64
    relaxed_binary_matrix::Array{Float64, 2}
end

function isTerminal(one_indices, zero_indices, k_sparse, matrix_dim)
    """
    This function evaluates whether the current node in the Branch and Bound
    tree is a terminal node.

    :param zero_indices: List of 2-tuples of Int64 where each entry consists of
                         an index (i, j) such that Y_ij is constrained to take
                         value 0.
    :param one_indices: List of 2-tuples of Int64 where each entry consists of
                        an index (i, j) such that S_ij is constrained to take
                        value 1 where S is the binary matrix denoting the
                        sparsity pattern of Y.
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param matrix_dim: The row and column dimension of the data (Int64).

    :return: True if this is a terminal node, false otherwise (Bool).
    """

    num_fixed_ones = size(one_indices)[1]
    num_fixed_zeros = size(zero_indices)[1]

    if num_fixed_ones == k_sparse
        return true
    end

    if num_fixed_zeros == matrix_dim ^ 2 - k_sparse
        return true
    end

    return false

end

function SLR_BnB(U, mu, lambda, k_sparse, k_rank; epsilon=0.1,
    iter_min_improv=1e-12, symmetric_tree=false, output_to_file=false,
    output_file_name="temp.txt")
    """
    This function computes a certifiably near-optimal solution to the problem
        min ||U - X - Y||_F^2 + lambda * ||X||_F^2 + mu * ||Y||_F^2
        subject to rank(X) <= k_rank, ||Y||_0 <= k_sparse

    by executing a custom branch and bound algorithm.

    :param U: An arbitrary n-by-n matrix.
    :param mu: The regularization parameter for the sparse matrix penalty
                   (Float64).
    :param lambda: The regularization parameter for the low rank matrix penalty
                   (Float64).
    :param k_sparse: The maximum sparsity of the sparse matrix (Int64).
    :param k_rank: The maximum rank of the low rank matrix (Int64).
    :param epsilon: The Branch and Bound termination criterion. This function
                    will terminate with an epsilon optimal solution to the
                    optimization problem (Float64).
    :param iter_min_improv: The minimal fractional decrease in the objective
                            value required for the alternating minimization
                            procedure to continue iterating.
    :param symmetric_tree: If true, Branch and Bound will only explore symmetric
                           sparsity patterns (Bool).
    :param output_to_file: If true, progress output will be printed to file. If
                           false, progress output will be printed to the console
                           (Bool).
    :param output_file_name: Name of the file progress output will be printed to
                             when output_to_file=true (String).

    :return: This function returns eight values:
             1) A tuple of two n-by-n arrays that correspond to the globally
                optimal solution to the optimization problem. The first element
                in the tuple is the optimal low rank matrix and the second
                element in the tuple is the optimal sparse matrix.
             2) The final global upper bound (Float64).
             3) The final global lower bound (Float64).
             4) The number of nodes explored during the optimization process
                (Int64).
             5) A list of values of type Float64 representing the evolution of
                the upper bound during the optimization procedure.
             6) A list of values of type Float64 representing the evolution of
                the lower bound during the optimization procedure.
             7) The total time elapsed during the optimiation process
                (milliseconds).
             8) The number of terminal nodes explored during the optimization
                process (Int64).

                best feasible solution found
             by this method (the first element in the tuple is the matrix X and
             the second element is the matrix Y). The second value is the
             objective value (Float64) achieved by the returned feasible
             solution.
    """

    return (global_X_sol, global_Y_sol),
           global_upper_bound,
           global_lower_bound,
           num_explored_nodes,
           upper_bound_hist,
           lower_bound_hist,
           total_time,
           terminal_nodes

    start_time = now()

    # Solve the root node
    init_zero_indices = []
    init_one_indices = []

    upper_bound_sol, upper_bound_obj = SLR_AM(U, mu, lambda, k_sparse, k_rank,
        random_restarts=0, zero_indices=init_zero_indices,
        one_indices=init_one_indices, min_improvement=iter_min_improv)

    lower_bound_sol, lower_bound_obj = solve_perspective_relaxation(U, mu,
        lambda, k_sparse, k_rank, zero_indices=init_zero_indices,
        one_indices=init_one_indices)

    root_optimality_gap = (upper_bound_obj - lower_bound_obj) / upper_bound_obj
    if output_to_file
        open(output_file_name, "a") do io
            write(io, "Root Node has been solved.\n")
            write(io, "Root Node upper bound is: $upper_bound_obj\n")
            write(io, "Root Node lower bound is: $lower_bound_obj\n")
            write(io, "Root Node optimality gap is: $root_optimality_gap\n\n")
        end
    else
        println("Root Node has been solved.")
        println("Root Node upper bound is: $upper_bound_obj")
        println("Root Node lower bound is: $lower_bound_obj")
        println("Root Node optimality gap is: $root_optimality_gap")
        println()
    end

    # Initialize the node list and global bounds
    master_node_list = []
    root_node = node(init_zero_indices,
                     init_one_indices,
                     upper_bound_obj,
                     lower_bound_obj,
                     lower_bound_sol[3])
    push!(master_node_list, root_node)
    global_upper_bound = upper_bound_obj
    global_lower_bound = lower_bound_obj
    global_X_sol = upper_bound_sol[1]
    global_Y_sol = upper_bound_sol[2]

    num_explored_nodes = 1
    upper_bound_hist = [global_upper_bound]
    lower_bound_hist = [global_lower_bound]

    n = size(U)[1]

    terminal_nodes = 0
    # Main branch and bound loop
    while (global_upper_bound - global_lower_bound) /
                                            global_upper_bound > epsilon

        """
        # Select the most recently added node (DFS)
        current_node = pop!(master_node_list)
        """
        current_node_index = findall(this_node->
            this_node.lower_bound == global_lower_bound, master_node_list)[1]
        current_node = master_node_list[current_node_index]
        deleteat!(master_node_list, current_node_index)

        # If the node is a terminal node, place it back on the master list
        if isTerminal(current_node.one_indices, current_node.zero_indices,
                      k_sparse, n)
            prepend!(master_node_list, current_node)
            continue
        end
        # Select entry to branch on
        index = argmin(abs.(current_node.relaxed_binary_matrix .- 0.5))
        i = index[1]
        j = index[2]

        # Construct the two child sets of indices from this parent node
        one_condition = size(current_node.one_indices)[1] <= k_sparse - 2
        zero_condition = size(current_node.zero_indices)[1] <= n^2 - k_sparse - 2
        sym_branch_feasible = one_condition && zero_condition
        if symmetric_tree && sym_branch_feasible && (i != j)
            new_index_zero = (append!(copy(current_node.zero_indices),
                              [(i, j), (j, i)]), copy(current_node.one_indices))
            new_index_one = (copy(current_node.zero_indices),
                             append!(copy(current_node.one_indices),
                             [(i, j), (j, i)]))
            new_index_list = [new_index_zero, new_index_one]
        else
            new_index_zero = (append!(copy(current_node.zero_indices), [(i, j)]),
                              copy(current_node.one_indices))
            new_index_one = (copy(current_node.zero_indices),
                             append!(copy(current_node.one_indices), [(i, j)]))
            new_index_list = [new_index_zero, new_index_one]
        end

        for (zero_indices, one_indices) in new_index_list

            upper_bound_sol, upper_bound_obj = SLR_AM(U, mu, lambda, k_sparse,
                        k_rank, random_restarts=0, zero_indices=zero_indices,
                        one_indices=one_indices,
                        min_improvement=iter_min_improv)

            if isTerminal(one_indices, zero_indices, k_sparse, n)
                lower_bound_sol = upper_bound_sol
                lower_bound_obj = upper_bound_obj
                relaxed_binary_matrix = abs.(lower_bound_sol[2]) .> 1e-10
                terminal_nodes = terminal_nodes + 1
            else
                lower_bound_sol, lower_bound_obj = solve_perspective_relaxation(
                    U, mu, lambda, k_sparse, k_rank, zero_indices=zero_indices,
                    one_indices=one_indices)
                relaxed_binary_matrix = lower_bound_sol[3]
            end

            if upper_bound_obj < global_upper_bound
                global_upper_bound = upper_bound_obj
                global_X_sol = upper_bound_sol[1]
                global_Y_sol = upper_bound_sol[2]

                current_gap = global_upper_bound - global_lower_bound
                current_gap = current_gap / global_upper_bound

                println("The new upper bound is: $global_upper_bound")
                println("The current lower bound is: $global_lower_bound")
                println("The current optimality gap is: $current_gap")
                println()

                if output_to_file
                    open(output_file_name, "a") do io
                        write(io, "The new upper bound is: $global_upper_bound\n")
                        write(io, "The current lower bound is: $global_lower_bound\n")
                        write(io, "The current optimality gap is: $current_gap\n\n")
                    end
                else
                    println("The new upper bound is: $global_upper_bound")
                    println("The current lower bound is: $global_lower_bound")
                    println("The current optimality gap is: $current_gap")
                    println()
                end

                new_master_node_list = []
                for this_node in master_node_list
                    if this_node.lower_bound < global_upper_bound
                        push!(new_master_node_list, this_node)
                    end
                end

                pre_size = size(master_node_list)[1] + 1
                post_size = size(new_master_node_list)[1] + 1

                if output_to_file
                    open(output_file_name, "a") do io
                        write(io, "Tree size before pruning: $pre_size\n")
                        write(io, "Tree size after pruning: $post_size\n\n")
                    end
                else
                    println("Tree size before pruning: $pre_size")
                    println("Tree size after pruning: $post_size")
                    println()
                end

                master_node_list = new_master_node_list
            end

            if lower_bound_obj < global_upper_bound
                new_node = node(zero_indices,
                                one_indices,
                                upper_bound_obj,
                                lower_bound_obj,
                                relaxed_binary_matrix)
                push!(master_node_list, new_node)
            end

            num_explored_nodes = num_explored_nodes + 1
            append!(upper_bound_hist, global_upper_bound)
            append!(lower_bound_hist, global_lower_bound)

            if num_explored_nodes % 200 == 0
                current_time = now() - start_time

                if output_to_file
                    open(output_file_name, "a") do io
                        write(io, "$num_explored_nodes nodes have been explored.\n")
                        write(io, "Current elapsed time: $current_time\n")
                        write(io, "The current lower bound is: $global_lower_bound\n")
                    end
                else
                    println("$num_explored_nodes nodes have been explored.")
                    println("Current elapsed time: $current_time")
                    println("The current lower bound is: $global_lower_bound")
                end
            end

        end

        if size(master_node_list)[1] == 0
            if output_to_file
                open(output_file_name, "a") do io
                    write(io, "All nodes have been explored")
                end
            else
                println("All nodes have been explored")
            end
            break
        end

        global_lower_bound = master_node_list[1].lower_bound
        for current_node in master_node_list
            if current_node.lower_bound < global_lower_bound
                global_lower_bound = current_node.lower_bound
            end
        end
    end

    total_time = now() - start_time

    if output_to_file
        open(output_file_name, "a") do io
            write(io, "An optimal solution has been found!")
        end
    else
        println("An optimal solution has been found!")
    end

    return (global_X_sol, global_Y_sol),
           global_upper_bound,
           global_lower_bound,
           num_explored_nodes,
           upper_bound_hist,
           lower_bound_hist,
           total_time,
           terminal_nodes

end;
