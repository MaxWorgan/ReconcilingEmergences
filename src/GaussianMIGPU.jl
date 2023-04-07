using CUDA
using LinearAlgebra: checksquare, BlasInt

"""
    TODO:
    1) avoid reshaping of arrays

"""
function computeMU(x, v, C, rho, c_index, col_means, col_std, logdet_rho, logdet_s2rho)

    @cuda threads = 100 calc_ψ_kernel(x, v, C, rho, c_index, col_means, col_std)

    reshaped_s1rho = [(rho[1, 1, i]) for i in 1:size(rho)[end]] |> cu
    reshaped_s2rho = [(rho[2:end, 2:end, i]) for i in 1:size(rho)[end]] 
    reshaped_rho   = [(rho[:, :, i]) for i in 1:size(rho)[end]]

    rho, _ = callBatchedCholesky(reshaped_rho)
    input1 = reshape(reduce(hcat, rho), 11, 11, size(x,2))
    @cuda threads = 100 logdet_kernel(input1, logdet_rho)

    s2rho, _ = callBatchedCholesky(reshaped_s2rho)
    input2 = reshape(reduce(hcat, s2rho), 10, 10, size(x,2))
    @cuda threads = 100 logdet_kernel(input2, logdet_s2rho)

    return sum(0.5 .* (log(reshaped_s1rho[1]).+ logdet_s2rho .- logdet_rho))

end

"""
    Calculate ψ across two time series X and V
    X is the time series1 - 48(minus last) x 18000
    V is the time series2 - 48(minus first) x 10
    C is the concatenated array - 47 x 11 x 18000
    rho is the correlation matrix - 11 x 11 x 18000
    col_means is the mean of each column of C - 11 x 18000

    The main indicies
"""
function calc_ψ_kernel(X, V, C, rho, c_index, col_means, col_std)
    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # horizontally concatenate A and B into C
    if thread_id <= size(X,2)
        A = @view(X[1:end-1, thread_id])
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

# inputs = 1D array of size batch containing 2D arrays of size (n, n) - the top diagonal is the cholesky factor
# outputs = 1D array of size batch containing the log det of each matrix
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

    return A, info

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
@inline function unsafe_batch(batch::Vector{<:CuArray{T}}) where {T}
    ptrs = pointer.(batch)
    return CuArray(ptrs)
end

num_of_samples = 18000

x = CUDA.rand(Float32, 48, num_of_samples) |> cu
v = CUDA.rand(Float32, 48, 10) |> cu
c_index = [CartesianIndices((1:(size(x,1) -1), 1:(size(v,2) + 1), i:i)) for i in 1:size(x)[end]] |> cu
logdet_rho = CUDA.zeros(Float32, size(x)[end])
logdet_s2rho = CUDA.zeros(Float32, size(x)[end])

C = CUDA.zeros(Float32, 47, 11, size(x)[end])
rho = CUDA.zeros(Float32, 11, 11, size(x)[end])
col_means = CUDA.zeros(size(x)[end], 11)
col_std = CUDA.zeros(size(x)[end], 11)
@btime computeMU(x,v[2:end, :], C, rho, c_index, col_means, col_std, logdet_rho, logdet_s2rho)


using BenchmarkTools
logdet_rho
logdet_s2rho

col_std
col_means

rho

C

CUSOLVER.potrfBatched!
