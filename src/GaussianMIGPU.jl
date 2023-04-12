using CUDA
using LinearAlgebra
using LinearAlgebra: BlasInt, checksquare
using Statistics 

function compute_single_MI(x,v,C,col_means,col_std,rho)

    @cuda threads = 512 blocks = 80 calculate_cor_kernel(x, v, C, col_means, col_std, rho)

    Dx = size(x,2)
    ld1 = logdet_via_LU(rho[1:Dx, 1:Dx])
    ld2 = logdet_via_LU(rho[Dx+1:end, Dx+1:end])
    ld3 = logdet_via_LU(rho)

    return sum(0.5 * (ld1 + ld2 - ld3))

end

function compute_reduced_MI(x, v, C, rho, c_index, col_means, col_std, logdet_rho, logdet_s2rho)

    @cuda threads = 384 blocks = 80 calculate_cor_kernel_batched(x, v, C, rho, c_index, col_means, col_std)

    reshaped_s1rho = CUDA.@allowscalar [(rho[1, 1, i]) for i in 1:size(rho)[end]]
    reshaped_s2rho = CUDA.@allowscalar [(rho[2:end, 2:end, i]) for i in 1:size(rho)[end]] 
    reshaped_rho   = CUDA.@allowscalar [(rho[:, :, i]) for i in 1:size(rho)[end]]

    rho    = callBatchedCholesky(reshaped_rho)
    input1 = reshape(reduce(hcat, rho), (11, 11, size(x,2)))
    @cuda threads = 1024 blocks = 160 logdet_kernel(input1, logdet_rho)

    s2rho = callBatchedCholesky(reshaped_s2rho)
    input2 = reshape(reduce(hcat, s2rho), 10, 10, size(x,2))
    @cuda threads = 1024 blocks=160 logdet_kernel(input2, logdet_s2rho)

    return CUDA.@allowscalar sum(0.5 .* (log(reshaped_s1rho[1]).+ logdet_s2rho .- logdet_rho))

end

"""

    Calculate the correlation matrix of X and V
    Step 1) Concatenate X and V into C
    Step 2) Calculate the mean of each column of C
    Step 3) Calculate the standard deviation of each column of C
    Step 4) Center the data by subtracting the mean and dividing by the standard deviation of each column of C 
    Step 5) Calculate the correlation matrix of C (by calculating the covariance matrix of the centered C)

"""
function calculate_cor_kernel(X,V,C,col_means,col_std,rho)

    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if thread_id <= (size(X,2) + size(V,2))
        c = thread_id
        if c <= size(X, 2) # if 51 <= 50
            for r in 1:size(X,1) # do a row
                C[r,c] = X[r, c]
            end
        else
            for r in 1:size(V,1) # do a row
                C[r,c] = V[r, c-size(X,2)]
            end
       end
    end

    sync_threads()

     # calculate the mean of each column of C
    if thread_id <= size(C,2)
        sum = 0.0f32
        for c_j in 1:size(C,1) ## 47
            sum += C[c_j, thread_id]
        end
        col_means[thread_id] = sum / size(C, 1)
    end

    sync_threads()

    # calculate the std of each column of C
    if thread_id <= size(C,2)
        sum = 0.0f32
        for c_j in 1:size(C)[1] ## 47
            sum += (C[c_j, thread_id] - col_means[thread_id])^2
        end
        col_std[thread_id] = sqrt(sum / (size(C, 1) - 1))
    end

    sync_threads()

    # center data by subtracting mean and dividing by std
    if thread_id <= size(C, 2)
        for c_i in 1:size(C, 1) ## 47
            C[c_i, thread_id] -= col_means[thread_id]
            C[c_i, thread_id] /= col_std[thread_id]
        end
    end

    sync_threads()

    # calculate covariance of the centered data correlation matrix
    if thread_id <= size(rho, 2)
        i = thread_id # 18010
        for j in 1:size(rho, 2) # 18010
            s = 0.0
            for k in 1:size(C, 1)
                s += C[k, i] * C[k, j]
            end
            rho[i, j] = s / (size(C, 1) - 1)
            rho[j, i] = rho[i, j]
        end
    end

    sync_threads()

    return nothing

end

"""
    Batch calculate the correlation matricies across two time series X and V
    X is the time series1 - n x m
    V is the time series2 - n x p
    C is the concatenated array - n x p+1 x m
    rho is the correlation matrix - p+1 x p+1 x m
    col_means is the mean of each column of C - p+1 x m
    col_std is the standard deviation of each column of C - p+1 x m

"""
function calculate_cor_kernel_batched(X, V, C, rho, c_index, col_means, col_std)
    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # horizontally concatenate A and B into C
    if thread_id <= size(X,2)
        A = @view(X[:, thread_id])
        cix = c_index[thread_id]
        for ci in cix
            if ci[2] <= size(A, 2)
                C[ci] = A[ci[1], ci[2]]
            elseif ci[2] <= size(V, 2) + 1
                C[ci] = V[ci[1], ci[2]-1]
            end
        end
    end

    sync_threads()

    # calculate the mean of each column of C
    if thread_id <= size(X,2)
        cix = c_index[thread_id]
        t_c = @view(C[cix])
        for c_i in 1:size(cix)[2] ## 11
            sum = 0.0f32
            for c_j in 1:size(cix)[1] ## 47
                sum += t_c[c_j, c_i]
            end
            col_means[thread_id, c_i] = sum / size(cix, 1)
        end
    end

    sync_threads()

    # calculate the std of each column of C
    if thread_id <= size(X,2)
        cix = c_index[thread_id]
        t_c = @view(C[cix])
        for c_i in 1:size(cix)[2] ## 11
            sum = 0.0f32
            for c_j in 1:size(cix)[1] ## 47
                sum += (t_c[c_j, c_i] - col_means[thread_id, c_i])^2
            end
            col_std[thread_id, c_i] = sqrt(sum / (size(cix, 1) - 1))
        end
    end

    sync_threads()



    # center data by subtracting mean and dividing by std
    if thread_id <= size(col_means, 1)
        cix = c_index[thread_id]
        t_c = @view(C[cix])
        for c_i in 1:size(cix, 1) ## 47
            for c_j in 1:size(cix, 2) ## 11
                t_c[c_i, c_j] -= col_means[thread_id, c_j]
                t_c[c_i, c_j] /= col_std[thread_id, c_j]
            end
        end
    end

    sync_threads()

    # calculate covariance of the centered data correlation matrix
    if thread_id <= size(rho, 3)
        cix = c_index[thread_id]
        t_c = @view(C[cix])
        v_rho = @view(rho[:, :, thread_id])
        for i in 1:size(rho, 1) # 11
            for j in 1:size(rho, 2) # 11
                s = 0.0
                for k in 1:size(t_c, 1)
                    s += t_c[k, i] * t_c[k, j]
                end
                v_rho[i, j] = s / (size(t_c, 1) - 1)
                v_rho[j, i] = v_rho[i, j]
            end
        end
    end

    sync_threads()

    return nothing
end

"""

    Batch calculate the log determinant of a cholesky decomposition of a correlation matrix
    inputs - the batch of cholesky decomposition of the correlation matricies

"""
function logdet_kernel(inputs, outputs)

    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    num_batches = size(inputs,3)
    input_size  = size(inputs,1)

    if thread_id <= num_batches
        tr = 0.0
        for i in 1:input_size
            tr += log(inputs[i, i, thread_id])
        end
        outputs[thread_id] = 2.0 * tr
    end

    return nothing
end

"""

    Calculate the log determinant of a single matrix, used for the v_mi calculation
    LU decomposition seems to be more stable than cholesky decomposition

"""
function logdet_via_LU(A)
    x,_ = CUSOLVER.getrf!(copy(A))
    return sum(log.(abs.(diag(x))))
end

"""

    Call the batch cholesky solver library - reimplemented here because the original couldn't handle views of matricies
    TODO: investigate if this is still true

"""
function callBatchedCholesky(A)
    # Set up information for the solver arguments
    n = checksquare(A[1])
    lda = max(1, stride(A[1], 2))
    batchSize = length(A)
    devinfo = CuArray{Cint}(undef, batchSize)

    Aptrs = unsafe_batch(A)

    # Run the solver
    CUSOLVER.cusolverDnSpotrfBatched(CUSOLVER.dense_handle(), 'U', n, Aptrs, lda, devinfo, batchSize)

    # Copy the solver info and delete the device memory
    info = CUDA.@allowscalar collect(devinfo)
    CUDA.unsafe_free!(devinfo)

    # Double check the solver's exit status
    for i = 1:batchSize
        chkargsok(BlasInt(info[i]))
    end

    return A

end

function chkargsok(ret::BlasInt)
    if ret < 0
        throw(ArgumentError("invalid argument #$(-ret) to LAPACK call"))
    end
end
# create a batch of pointers in device memory from a batch of device arrays
@inline function unsafe_batch(batch::Vector{<:CuArray{T}}) where {T}
    ptrs = pointer.(batch)
    return CuArray(ptrs)
end