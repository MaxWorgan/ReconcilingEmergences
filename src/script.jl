using CUDA
using JLD
using Statistics
using LinearAlgebra
using DSP

const threads_per_block = 256

function gpu_mutual_information(x::CuArray{T}, y::CuArray{T}, kernel_size::Int, sigma::Float32) where {T<:AbstractFloat}
    N = size(x)[1]

    kernel_x = DSP.gaussian(kernel_size, sigma) |> cu
    kernel_y = DSP.gaussian(kernel_size, sigma) |> cu

    threads_per_block = 256
    blocks = ceil(Int, N / threads_per_block)

    # Allocate memory for convolved arrays on the device
    conv_x = CuArray{Float32}(undef, N)
    conv_y = CuArray{Float32}(undef, N)

    # Perform convolution and sum on the GPU
    @cuda threads = threads_per_block blocks = blocks convolve_and_sum_kernel(conv_x, conv_y, x, y, kernel_x, kernel_y, N)

    # Compute mean and standard deviation of convolved arrays
    μ_x = mean(conv_x)
    μ_y = mean(conv_y)
    σ_x = std(conv_x)
    σ_y = std(conv_y)

    # Allocate memory for joint probability distribution on the device
    joint_prob = CuArray{Int64}(undef, kernel_size, kernel_size)

    # Compute joint probability distribution on the GPU
    @cuda threads = threads_per_block blocks = blocks joint_prob_kernel(joint_prob, conv_x, conv_y, μ_x, μ_y, σ_x, σ_y, kernel_size, N)

    # Compute marginal probability distributions on the GPU
    marginal_x = sum(joint_prob, dims=2)
    marginal_y = sum(joint_prob, dims=1)

    # Compute mutual information on the GPU
    @cuda threads = threads_per_block blocks = blocks mutual_information_kernel(joint_prob, marginal_x, marginal_y, N)

    # Copy mutual information result back to host
    mi = CuArray{Float32}(undef, 1)
    # CUDA.@sync CUDA.(mi, joint_prob[1], sizeof(Float32))

    return mi[1]
end


function convolve_and_sum_kernel(conv_x, conv_y, x, y, kernel_x, kernel_y, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        conv_x[i] = DSP.conv(x[:, i], kernel_x)[1]
        conv_y[i] = DSP.conv(y[:, i], kernel_y)[1]
    end
    return nothing
end

function joint_prob_kernel(joint_prob, conv_x, conv_y, μ_x, μ_y, σ_x, σ_y, kernel_size, N)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        x_index = max(1, min(kernel_size, Int(round(((conv_x[i] - μ_x) / σ_x) * (kernel_size / 6) + kernel_size / 2))))
        y_index = max(1, min(kernel_size, Int(round(((conv_y[i] - μ_y) / σ_y) * (kernel_size / 6) + kernel_size / 2))))
        atomicAdd!(joint_prob[x_index, y_index], 1)
    end
    return nothing
end

function cov_kernel(x::CuArray, y::CuArray, out::CuArray)
    # Determine the number of columns in the concatenated matrix
    n_cols = size(x, 2) + size(y, 2)

    # Determine the number of rows in the matrix
    n_rows = size(x, 1)

    # Determine the thread index and block index within the GPU grid
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # Determine the column index within the concatenated matrix
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # Calculate the covariance matrix element for this thread
    if i <= n_cols && j <= n_cols
        if i <= size(x, 2) && j <= size(x, 2)
            # Covariance matrix element between x and x
            out[i, j] = cov(x[:, i], x[:, j])
        elseif i > size(x, 2) && j > size(x, 2)
            # Covariance matrix element between y and y
            out[i, j] = cov(y[:, i-size(x, 2)], y[:, j-size(x, 2)])
        else
            # Covariance matrix element between x and y
            out[i, j] = cov([x[:, i-size(x, 2)]; y[:, j]])
        end
    end

    return nothing
end


function cov_kernel(A::CuArray{Float32,1}, B::CuArray{Float32,2})
    # get size of input arrays
    m = size(A)[1]
    n1 = 1
    _, n2 = size(B)

    # concatenate input arrays horizontally
    C = CUDA.zeros(Float32, m, n1 + n2)
    C[:, 1:n1] .= A
    C[:, n1+1:end] .= B

    # calculate covariance matrix
    D = CUDA.zeros(Float32, n1 + n2, n1 + n2)
    for i in 1:(n1+n2)
        for j in i:(n1+n2)
            D[i, j] = CUDA.sum(C[:, i] .* C[:, j]) / m
            D[j, i] = D[i, j]
        end
    end

    return D
end

function hcat_k!(C::CuArray, A::CuArray, B::CuArray)
    for i in 1:size(A, 1)
        for j in 1:size(A, 2)
            C[i, j] = A[i, j]
        end
        for j in 1:size(B, 2)
            C[i, size(A, 2)+j] = B[i, j]
        end
    end
    return nothing
end

function cov_gpu!(out::CuArray, A::CuArray, B::CuArray)
    # horizontally concatenate A and B
    C = CUDA.zeros(size(A, 1), size(A, 2) + size(B, 2))
    @cuda threads = (size(A, 2) + size(B, 2))
    for i in 1:size(A, 1) #47
        C[i, j] = A[i, j]
        for j in 1:size(B, 2)
            C[i, size(A, 2)+j] = B[i, j]
        end
    end

    # calculate the covariance matrix of C
    n = size(C, 2)
    C .-= mean(C, dims=2)
    cov_mat = (C * C') / (n - 1)

    # copy result to output
    out .= cov_mat

    return out
end



out = CUDA.zeros(size(A, 2) + size(B, 2), size(A, 2) + size(B, 2))
@cuda threads = (size(A, 1), size(B, 1)) cov_gpu!(out, A, B)




function hcat_gpu(A::CuArray, B::CuArray)
    # allocate output array on device
    C = CUDA.zeros(size(A, 1), size(A, 2) + size(B, 2))

    # compute grid and block dimensions
    blockdim = (32, 32)
    griddim = ((size(A, 2) + size(B, 2) - 1) ÷ blockdim[1] + 1, (size(A, 1) - 1) ÷ blockdim[2] + 1)

    # launch kernel
    @cuda threads = blockdim griddim = griddim hcat_kernel(C, A, B)

    return C
end


function cov_kernel(covA, centeredA)
    # compute global indices
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    # compute covariance elements
    if i <= j <= size(centeredA, 2)
        s = 0.0
        for k in 1:size(centeredA, 1)
            s += centeredA[k, i] * centeredA[k, j]
        end
        covA[i, j] = s / (size(centeredA, 1) - 1)
        covA[j, i] = covA[i, j]
    end

    return nothing
end

function gpu_gaussion(A, B)
    C = CUDA.zeros(Float32, (47, 11))
    rho = CUDA.zeros(11, 11)
    # allocate output array on device
    @cuda threads = 47 hcat_kernel(C, A, B)
    meanC = CUDA.sum(C, dims=1) / size(C, 1)
    # center data by subtracting mean
    centeredA = C .- meanC
    @cuda threads = (16, 16) cov_kernel(rho, centeredA)
    s1 = SubArray(rho, (1:1, 1:1))
    s2 = SubArray(rho, (2:11, 2:11))
    mi = 0.5 * (culogdet(s1) + culogdet(s2) - culogdet(rho))
    CUDA.unsafe_free!(rho)
    CUDA.unsafe_free!(C)
    return mi
end

culogdet(m) = logdet(cholesky(m, check=false))


using Folds, FoldsCUDA, LinearAlgebra, FLoops


function testeroo(x, v)
    v2 = v[]
    function test2(k)
        x2 = SubArray(x, (1:47, k))
        gpu_gaussion(x2, v2)
    end
    Folds.mapreduce(test2, +, 1:size(x, 2), CUDAEx())
end

try
    testeroo(x, v)
catch err
    CUDA.code_typed(err)
end




using StaticArrays

function hcat_kernel(C, A, B)
    # compute global indices
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # horizontally concatenate A and B
    if i <= size(A, 1)
        @inbounds C[i, 1] = A[i]
    end
    if i <= size(B, 1)
        for k in 1:size(B, 2)
            @inbounds C[i, k+1] = B[i, k]
        end
    end

    return nothing
end


# function centered_kernel(C)
#     i = (blockIdx().x-1)*blockDim().x+threadIdx().x


#original data into constant memory? - maybe not implemented in CUDA.jl
#use shared memory for intermediate results? can't use this with Dynamic Parallelism

#
# X = 48 x 100
# V = 48 x 10
"""
    Calculate ψ across two time series X and V
    X is the time series1 - 48(minus last) x 18000
    V is the time series2 - 48(minus first) x 10
    C is the concatenated array - 47 x 11 x 18000
    rho is the correlation matrix - 11 x 11 x 18000
    col_means is the mean of each column of C - 11 x 18000

    The main indicies
"""
function calc_ψ_kernel(X, V, C, rho, c_index, col_means)
    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    # cis = @view(c_idx[c_range_l:c_range_h])

    # horizontally concatenate A and B into C -- WORKING
    if thread_id < length(C)
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

    # calculate the mean of each column of C -- WORKING
    if thread_id < length(C)
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


    # center data by subtracting mean
    if thread_id <= size(col_means, 1)
        cix = c_index[thread_id]
        t_c = @view(C[cix])
        for c_i in 1:size(cix, 1) ## 47
            for c_j in 1:size(cix, 2) ## 11
                t_c[c_i, c_j] -= col_means[thread_id, c_j]
            end
        end
    end

    # calculate standard deviation of each column of C
    sync_threads()
    if thread_id <= size(col_means, 1)
        cix = c_index[thread_id]
        t_c = @view(C[cix])
        for c_i in 1:size(cix, 1) ## 47
            for c_j in 1:size(cix, 2) ## 11
                t_c[c_i, c_j] -= col_means[thread_id, c_j]
            end
        end
    end

    sync_threads()

    if thread_id <= size(rho, 3)
        cix = c_index[thread_id]
        t_c = @view(C[cix])
        v_rho = @view(rho[:, :, thread_id])
        for i in 1:size(rho, 1)
            for j in 1:size(rho, 2)
                # compute covariance matricies
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

function computeMU(x, v, C, rho, c_index, col_means, logdet_rho, logdet_s2rho)

    @cuda threads = 100 calc_ψ_kernel(x, v, C, rho, c_index, col_means)

    reshaped_s1rho = [@views(rho[1, 1, i]) for i in 1:size(rho)[end]] |> cu
    reshaped_s2rho = [@views(rho[2:end, 2:end, i]) for i in 1:size(rho)[end]]
    reshaped_rho = [@views(rho[:, :, i]) for i in 1:size(rho)[end]]

    rho, _ = callBatchedCholesky(reshaped_rho) # could be optimised to reuse memory each calculation
    s2rho, _ = callBatchedCholesky(reshaped_s2rho) # could be optimised to reuse memory each calculation

    @cuda threads = 100 logdet_kernel(reshape(reduce(hcat, rho), 11, 11, 100), logdet_rho)

    @cuda threads = 100 logdet_kernel(reshape(reduce(hcat, s2rho), 10, 10, 100), logdet_s2rho)

    return 0.5 .* (reshaped_s1rho .+ logdet_rho .- logdet_s2rho)

    # return (reshaped_s1rho, logdet_rho, logdet_s2rho)


end

using LinearAlgebra: checksquare, BlasInt

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
@inline function unsafe_batch(batch::Vector{T}) where {T}
    ptrs = pointer.(batch)
    return CuArray(ptrs)
end

C = CUDA.zeros(Float32, 47, 11, size(x)[end])
rho = CUDA.zeros(Float32, 11, 11, size(x)[end])
col_means = CUDA.zeros(size(x)[end], 11)
c_index = [CartesianIndices((1:47, 1:11, i:i)) for i in 1:size(x)[end]] |> cu

logdet_rho = CUDA.zeros(Float32, size(x)[end])
logdet_s2rho = CUDA.zeros(Float32, size(x)[end])
@btime computeMU(x, vv)

# inputs = 1D array of size batch containing 2D arrays of size (n, n) - the top diagonal is the cholesky factor
# outputs = 1D array of size batch containing the log det of each matrix
function logdet_kernel(inputs, outputs)

    thread_id = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    n = size(inputs, 1)
    tr = 0.0
    for i = 1:n
        tr += log(inputs[i, i, thread_id])
    end
    outputs[thread_id] = 2.0 * tr

    return nothing
end

@cuda threads = 100 logdet_kernel(inputs, outputs)

@cuda threads = 100 calc_ψ_kernel(x, vv, C, rho, c_index, col_means)

rho

reshaped_rho = [@views(rho[:, :, i]) for i in 1:size(rho)[end]]

rho, info = CUSOLVER.potrfBatched!('U', reshaped_rho)

using BenchmarkTools

x = CUDA.rand(Float32, 48, 100) |> cu

v = CUDA.rand(Float32, 48, 10) |> cu
#global memory used for correlation matrix
C = CUDA.zeros(Float32, 47, 11, size(x)[end]) |> cu

rho = CUDA.zeros(Float32, 11, 11, size(x)[end]) |> cu
c_index = [CartesianIndices((1:47, 1:11, i:i)) for i in 1:size(x)[end]] |> cu


col_means = CUDA.zeros(size(x)[end], 11)
#global memory used for output

# function calculate_indicies(x)
#     r_main  = ntuple(i->i == ndims(x) ? Base.OneTo(1) : axes(x)[i], ndims(x)) |> CartesianIndices
#     r_batch = ntuple(i->i != ndims(x) ? Base.OneTo(1) : axes(x)[i], ndims(x)) |> CartesianIndices
#     return r_main, r_batch
# end


# r_main, r_batch = calculate_indicies(x)

# c_main, c_batch = calculate_indicies(C)



vv = v[2:end, :]
@btime CUDA.@sync @cuda threads = 100 calc_ψ_kernel(o, x, vv, C, rho, c_index)

@cuda threads = 100 calc_ψ_kernel(o, x, vv, C, rho, c_index, rho_index, col_means)



col_means

C

Z = [x[1:end-1, 1] v[2:end, :]]

C[:, :, 1]

Z