using CUDA
# using NVTX

function EmergencePsiGPU(x,v, tau=1)

    # NVTX.@range "Allocating all the data" begin
        c_index      = [CartesianIndices((1:(size(x,1) -tau), 1:(size(v,2) + tau), i:i)) for i in 1:size(x)[end]] |> cu
        logdet_rho   = CUDA.zeros(Float32, size(x)[end])
        logdet_s2rho = CUDA.zeros(Float32, size(x)[end])

        C1         = CUDA.zeros(Float32, size(x,1)-1, size(v,2)+1, size(x)[end])
        rho1       = CUDA.zeros(Float32, size(C1,2), size(C1,2), size(x)[end])
        rho2       = CUDA.zeros(Float32, size(C1,2)-1, size(C1,2)-1, size(x)[end])
        col_means1 = CUDA.zeros(size(x)[end], 11)
        col_std1   = CUDA.zeros(size(x)[end], 11)

        C2         = CUDA.zeros(Float32, size(v,1)-1, size(v,2)*2)
        rho3       = CUDA.zeros(Float32, size(C2,2), size(C2,2))
        col_means2 = CUDA.zeros(size(C2)[2])
        col_std2   = CUDA.zeros(size(C2)[2])
    # end

    # NVTX.@range "Computing X MI" begin
        x_mi = compute_reduced_MI(x[1:end-tau,:], v[1+tau:end,:], C1, rho1, rho2, c_index, col_means1, col_std1, logdet_rho, logdet_s2rho)
    # end

    # NVTX.@range "Computing V MI" begin
        v_mi = compute_single_MI(v[1:end-tau,:], v[1+tau:end,:], C2, col_means2, col_std2, rho3)
    # end

    return (v_mi - x_mi, x_mi, v_mi)

end

function ChainRulesCore.rrule(::typeof(mcor), x::AbstractMatrix)
    pre = 1/(size(x,1)-1)
    xcorr = (x .- mean(x, dims=1)) ./ std(x, dims=1)
    z = pre * adjoint(xcorr) * xcorr  # right for complex
    function mcov_pullback(ȳ)
        return (NoTangent(), Hermitian(unthunk(ȳ)) * (xcorr) * 2pre)
    end
    return z, (δy -> mcov_pullback(δy))
end

function ChainRulesCore.rrule(::typeof(mcov), x::AbstractMatrix{T}; corrected=true) where {T}
    pre = one(float(T))/(size(x,2) - corrected)
    xcorr = x .- mean(x, dims=2)
    z = pre * xcorr * adjoint(xcorr)
    z, dz -> (NoTangent(), Hermitian(unthunk(dz)) * xcorr * 2pre)  # not sure about complex case
end


trace(M) = sum(diag(M))


function dmi_dx(Xμ, Vμ, XC, VC, XVC)
    Σxx_inv = inv(XC)
    Σxv = XVC[1:size(Xμ, 1), size(Xμ, 1)+1:end]
    Σvv_inv = inv(VC[size(Xμ, 1)+1:end, size(Xμ, 1)+1:end])
    Σvx = XVC[size(Xμ, 1)+1:end, 1:size(Xμ, 1)]
    Σvx_Sigma_xx_inv = Σvx * Σxx_inv
    
    term1 = -0.5 * trace((Σxx_inv - Σvx_Sigma_xx_inv * Σxv) * XΣ)
    term2 = -0.5 * trace(Σvv_inv * Σvx_Sigma_xx_inv * Σxv * Σxx_inv)
    term3 = 0.5 * trace(Σvv_inv * Σvx * Σxx_inv * Σvx_Sigma_xx_inv * XΣ)
    
    return term1 + term2 + term3
end

function inverse(M,I)

    CUSOLVER.getrf!(M)
    CUBLAS.cublasSgetria


end


function batched_inverse(M, I)
    _,n,batchSize = size(M)
    pivotArray = CuArray{Cint}(undef, (n, batchSize))
    devinfo = CuArray{Cint}(undef, batchSize)
    A = CuArray(map(pointer, eachslice(M, dims=3)))
    B = CuArray(map(pointer, eachslice(I, dims=3)))

    CUBLAS.cublasSgetrfBatched(CUBLAS.handle(), n, A, n, pivotArray, devinfo, batchSize)
    info = CUDA.@allowscalar collect(devinfo)
    for i = 1:batchSize
        if info[i] < 0
            throw(ArgumentError("invalid argument #$(-info[i]) to LAPACK call"))
        end
    end

    CUBLAS.cublasSgetriBatched(CUBLAS.handle(), n, A, n, pivotArray, B, n, devinfo, batchSize)
    info = CUDA.@allowscalar collect(devinfo)
    for i = 1:batchSize
        if info[i] < 0
            throw(ArgumentError("invalid argument #$(-info[i]) to LAPACK call"))
        end
    end

    CUDA.unsafe_free!(devinfo)
    CUDA.unsafe_free!(pivotArray)

    return nothing

end

function δψ(Xμ, Vμ, XC, VC, C)
    



end


"""

x =  1 2 3
     4 5 6
     7 8 9

y =  1 2 3
     4 5 6

"""
function cross_cor_kernel(x, y, xμ, yμ, o)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if idx <= size(x, 1) && idy <= size(y, 1)
        o[idx, idy] = (x[idx, idy] - xμ[idx]) * (y[idx, idy] - yμ[idy])
    end
  




    



end



function dmi_dx_and_dy_corr(X::AbstractMatrix, Y::AbstractMatrix, mean_X::AbstractVector, mean_Y::AbstractVector, Σxx_corr::AbstractMatrix, Σyy_corr::AbstractMatrix, Σxy_corr::AbstractMatrix)
    # Compute the inverse of the covariance matrices
    Σxx_corr_inv = inv(Σxx_corr)
    Σyy_corr_inv = inv(Σyy_corr)

    # Compute the cross-covariance matrix
    cc = crosscov(X, Y)

    Σxy = reshape(cc, (size(X,1)*2-1), size(Y,2))
    # Compute the inverse of the cross-covariance matrix
    Σxy_inv = pinv(Σxy)

    @show Σxx_corr_inv |> size
    @show Σxy_inv |> size
    @show Σxy_corr |> size
    @show Σxx_corr_inv |> size
    @show mean_X |> size
    # Compute the first term of the derivative with respect to X
    term1_x = (Σxx_corr_inv * mean_X') - (Σxy_inv * Σxy_corr * Σxx_corr_inv * mean_X')

    # Compute the second term of the derivative with respect to X
    term2_x = -0.5 * Σxx_corr_inv * (trace(Σxx_corr_inv * Σxy_corr * Σyy_corr_inv * Σxy_corr') - trace(Σxy_inv * Σyy_corr_inv * Σxy_corr' * Σxx_corr_inv * Σxy_corr * Σyy_corr_inv))

    # Compute the first term of the derivative with respect to Y
    term1_y = -Σyy_corr_inv * mean_Y + Σxy_inv * Σxy_corr * Σyy_corr_inv * mean_Y

    # Compute the second term of the derivative with respect to Y
    term2_y = -0.5 * Σyy_corr_inv * (trace(Σyy_corr_inv * Σxy_corr' * Σxx_corr_inv * Σxy_corr * Σyy_corr_inv) - trace(Σxy_inv * Σxx_corr_inv * Σxy_corr' * Σyy_corr_inv * Σxy_corr * Σxx_corr_inv))

    return term1_x + term2_x, term1_y + term2_y
end


using LinearAlgebra

function dmi_dx_and_dy(X::AbstractMatrix, Y::AbstractMatrix, Xμ::AbstractVector, Yμ::AbstractVector)
    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)

    # Compute cross-covariance matrix of X and Y
    cc = crosscov(X,Y)
    Σxy = reshape(cc, (n*2-1), q) 

    # Compute inverse of stacked covariance matrix
    Σinv = pinv(Σxy)

    # Compute derivatives with respect to X
    dmi_dx = zeros(n, p)
    for i = 1:n
        dmi_dx[i, :] = -0.5 .* (X[i, :] .- Xμ) * Σinv[1:p, 1:p]
        for j = 1:q
            dmi_dx[i, :] += Σinv[1:p, p+j] .* (Y[i, j] .- Yμ[j])
        end
    end

    # Compute derivatives with respect to Y
    dmi_dy = zeros(n, q)
    for i = 1:n
        dmi_dy[i, :] = -0.5 .* (Y[i, :] .- Yμ) * Σinv[p+1:end, p+1:end]
        for j = 1:p
            dmi_dy[i, :] += Σinv[p+j, p+1:end] .* (X[i, j] .- Xμ[j])
        end
    end

    return dmi_dx, dmi_dy
end

using LinearAlgebra

function dmi_dx_and_dy_correlation(X::Array{T,2}, Y::Array{T,2}) where T<:Real
    # X: n x p matrix
    # Y: n x q matrix

    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)

    # Compute correlation matrices
    Σxx = cor(X, dims=1) # p x p matrix
    Σyy = cor(Y, dims=1) # q x q matrix
    Σxy = cor(X, Y, dims=1) # p x q matrix
    Σyx = Σxy' # q x p matrix

    # Compute inverse of covariance matrices
    Σxx_inv = inv(Σxx) # p x p matrix
    Σyy_inv = inv(Σyy) # q x q matrix

    # Compute cross-covariance matrices
    Σyx_Σyy_inv = Σyx' * Σyy_inv # q x p matrix
    Σyx_Σxx_inv = Σyx * Σxx_inv' # q x p matrix
    Σxy_Σxx_inv = Σxy' * Σxx_inv # p x q matrix
    Σxy_Σyy_inv = Σxy * Σyy_inv' # p x q matrix

    # Compute derivatives
    dI_dx = -0.5 .* (Σxx_inv .* (Σxy_Σyy_inv * Σyx_Σyy_inv') .+ Σxy_Σxx_inv') # p x p matrix
    dI_dy = -0.5 .* (Σyy_inv .* (Σyx_Σxx_inv * Σxy_Σxx_inv') .+ Σyx_Σyy_inv') # q x q matrix

    return dI_dx, dI_dy
end


function dmi_dx_and_dy_correlation(X::Array{T,2}, Y::Array{T,2}) where T<:Real
    # X: n x p matrix
    # Y: n x q matrix

    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)

    # Compute correlation matrices
    Σxx = cor(X, dims=1) # p x p matrix
    Σyy = cor(Y, dims=1) # q x q matrix
    Σxy = cor(X, Y, dims=1) # p x q matrix
    Σyx = Σxy' # q x p matrix

    # Compute inverse of covariance matrices
    Σxx_inv = inv(Σxx) # p x p matrix
    Σyy_inv = inv(Σyy) # q x q matrix

    # Compute cross-covariance matrices
    Σyx_Σyy_inv = Σyx' * Σyy_inv # q x p matrix
    Σyx_Σxx_inv = Σyx * Σxx_inv' # q x p matrix
    Σxy_Σxx_inv = Σxy' * Σxx_inv # p x q matrix
    Σxy_Σyy_inv = Σxy * Σyy_inv' # p x q matrix

    # Compute derivatives
    dI_dx = Array{Float64}(undef, p, p) # p x p matrix
    dI_dy = Array{Float64}(undef, q, q) # q x q matrix

    for i in 1:p, j in 1:p
        @show -0.5 * (Σxx_inv[i, j] * (Σxy_Σyy_inv[:, i]' * Σyx_Σyy_inv[:, j]) + Σxy_Σxx_inv[i, j])
    end

    for i in 1:q, j in 1:q
        @show  -0.5 * (Σyy_inv[i, j] * (Σyx_Σxx_inv[:, i]' * Σxy_Σxx_inv[:, j]) + Σyx_Σyy_inv[i, j])
    end

    return dI_dx, dI_dy
end


function dmi_dx_and_dy_correlation(X::Array{T,2}, Y::Array{T,2}) where T<:Real
    # X: n x p matrix
    # Y: n x q matrix

    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)

    # Compute correlation matrices
    Σxx = cor(X, dims=1) # p x p matrix
    Σyy = cor(Y, dims=1) # q x q matrix
    Σxy = cor(X, Y, dims=1) # p x q matrix
    Σyx = Σxy' # q x p matrix

    # Compute inverse of covariance matrices
    Σxx_inv = inv(Σxx) # p x p matrix
    Σyy_inv = inv(Σyy) # q x q matrix

    # Compute cross-covariance matrices
    Σyx_Σyy_inv = Σyx * Σyy_inv # q x p matrix
    Σyx_Σxx_inv = Σyx' * Σxx_inv # q x p matrix
    Σxy_Σxx_inv = Σxy * Σxx_inv # p x q matrix
    Σxy_Σyy_inv = Σxy' * Σyy_inv # p x q matrix

    # Compute derivatives
    dI_dx = -0.5 .* (Σxx_inv .* (Σxy_Σyy_inv * Σyx_Σyy_inv') .+ Σxy_Σxx_inv') # p x p matrix
    dI_dy = -0.5 .* (Σyy_inv .* (Σyx_Σxx_inv' * Σxy_Σxx_inv) .+ Σyx_Σyy_inv') # q x q matrix

    return dI_dx, dI_dy
end


function dmi_dx_and_dy_correlation(X::Array{T,2}, Y::Array{T,2}) where T<:Real
    # X: n x p matrix
    # Y: n x q matrix

    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)

    # Compute correlation matrices
    Σxx = cor(X, dims=1) # p x p matrix
    Σyy = cor(Y, dims=1) # q x q matrix
    Σxy = cor(X, Y, dims=1) # p x q matrix
    Σyx = Σxy' # q x p matrix

    # Compute inverse of covariance matrices
    Σxx_inv = inv(Σxx) # p x p matrix
    Σyy_inv = inv(Σyy) # q x q matrix

    # Compute cross-covariance matrices
    Σyx_Σyy_inv = Σyx' * Σyy_inv # q x p matrix
    Σyx_Σxx_inv = Σyx * Σxx_inv' # q x p matrix
    Σxy_Σxx_inv = Σxy' * Σxx_inv # p x q matrix
    Σxy_Σyy_inv = Σxy * Σyy_inv' # p x q matrix

    # Compute derivatives
    dI_dx = -0.5 .* (Σxx_inv .* (Σxy_Σyy_inv * Σyx_Σyy_inv') .+ Σxy_Σxx_inv') # p x p matrix
    dI_dy = -0.5 .* (Σyy_inv .* (Σyx_Σxx_inv' * Σxy_Σxx_inv) .+ Σyx_Σyy_inv') # q x q matrix
    
    @show dI_dx
    @show dI_dy
    # Compute gradients for each element of matrices X and Y
    dI_dx_all = zeros(T, n, p)
    dI_dy_all = zeros(T, n, q)
    
    for i in 1:n
        dI_dx_all[i, :, :] = X[i, :]' * dI_dx * X[i, :]
        dI_dy_all[i, :, :] = Y[i, :]' * dI_dy * Y[i, :]
    end

    return dI_dx_all, dI_dy_all
end


function dmi_dx_and_dy_correlation(X::Array{T,2}, Y::Array{T,2}) where T<:Real
    # X: n x p matrix
    # Y: n x q matrix

    n = size(X, 1)
    p = size(X, 2)
    q = size(Y, 2)

    # Compute correlation matrices
    Σxx = cor(X, dims=1) # p x p matrix
    Σyy = cor(Y, dims=1) # q x q matrix
    Σxy = cor(X, Y, dims=1) # p x q matrix
    Σyx = Σxy' # q x p matrix

    # Compute inverse of covariance matrices
    Σxx_inv = inv(Σxx) # p x p matrix
    Σyy_inv = inv(Σyy) # q x q matrix

    # Compute cross-covariance matrices
   Σyx_Σyy_inv = Σyx' * Σyy_inv # q x p matrix
   Σyx_Σxx_inv = Σyx * Σxx_inv' # q x p matrix
   Σxy_Σxx_inv = Σxy' * Σxx_inv # p x q matrix
   Σxy_Σyy_inv = Σxy * Σyy_inv' # p x q matrix

    # Compute derivatives
    dI_dx = zeros(T, n, p)
    dI_dy = zeros(T, n, q)
    for i in 1:n
        for j in 1:p
            println(-0.5 .* (Σxx_inv .* (Σxy_Σyy_inv * Σyx_Σyy_inv') .+ Σxy_Σxx_inv'))
        end
    end
    for i in 1:n
        for j in 1:q
            print(-0.5 .* (Σyy_inv .* (Σyx_Σxx_inv * Σxy_Σxx_inv') .+ Σyx_Σyy_inv'))
        end
    end

    return dI_dx, dI_dy
end

function gaussian_mutual_information_derivatives(X,Y)
    # X: T x A matrix
    # Y: T x B matrix

    T, A = size(X)
    _, B = size(Y)

    # Compute covariance matrices
    Σxx = cov(X, dims=1) # A x A matrix
    Σyy = cov(Y, dims=1) # B x B matrix
    Σxy = cov(X, Y, dims=1) # A x B matrix
    Σyx = Σxy' # B x A matrix

    # Compute inverse of covariance matrices
    Σxx_inv = inv(Σxx) # A x A matrix
    Σyy_inv = inv(Σyy) # B x B matrix

    # Compute cross-covariance matrices
    Σyx_Σyy_inv = Σyx' * Σyy_inv # B x A matrix
    Σyx_Σxx_inv = Σyx * Σxx_inv' # B x A matrix
    Σxy_Σxx_inv = Σxy' * Σxx_inv # B x A matrix
    Σxy_Σyy_inv = Σxy * Σyy_inv' # A x B matrix

    # Compute derivatives
    dI_dx = -0.5 .* (Σxx_inv .* (Σxy_Σyy_inv' * Σyx_Σyy_inv) .+ Σxy_Σxx_inv) # T x A matrix
    dI_dy = -0.5 .* (Σyy_inv .* (Σyx_Σxx_inv' * Σxy_Σxx_inv) .+ Σyx_Σyy_inv) # T x B matrix

    return dI_dx, dI_dy
end


function gaussian_mutual_information_derivatives(X,Y)
    # X: T x A matrix
    # Y: T x B matrix

    T, A = size(X)
    _, B = size(Y)

    # Compute covariance matrices
    Σxx = cov(X, dims=1) # A x A matrix
    Σyy = cov(Y, dims=1) # B x B matrix
    Σxy = cov(X, Y, dims=1) # A x B matrix
    Σyx = Σxy' # B x A matrix

    # Compute inverse of covariance matrices
    Σxx_inv = inv(Σxx) # A x A matrix
    Σyy_inv = inv(Σyy) # B x B matrix

    # Compute cross-covariance matrices
    Σyx_Σyy_inv = Σyx' * Σyy_inv # B x A matrix
    Σyx_Σxx_inv = Σyx * Σxx_inv' # B x A matrix
    Σxy_Σxx_inv = Σxy' * Σxx_inv # B x A matrix
    Σxy_Σyy_inv = Σxy * Σyy_inv' # A x B matrix

    # Compute derivatives
    dI_dx = -0.5 .* (Σxx_inv .* (Σxy_Σyy_inv * Σyx_Σyy_inv') .+ Σxy_Σxx_inv) # T x A matrix
    dI_dy = -0.5 .* (Σyy_inv .* (Σyx_Σxx_inv * Σxy_Σxx_inv') .+ Σyx_Σyy_inv) # T x B matrix

    return dI_dx, dI_dy
end


function gaussian_mutual_information_derivatives(X,Y)
    # X: T x A matrix
    # Y: T x B matrix

    T, A = size(X)
    _, B = size(Y)

    # Compute covariance matrices
    Σxx = cov(X, dims=1) # A x A matrix
    Σyy = cov(Y, dims=1) # B x B matrix
    Σxy = cov(X, Y, dims=1) # A x B matrix
    Σyx = Σxy' # B x A matrix

    # Compute inverse of covariance matrices
    Σxx_inv = inv(Σxx) # A x A matrix
    Σyy_inv = inv(Σyy) # B x B matrix

    # Compute cross-covariance matrices
    Σyx_Σyy_inv = Σyx' * Σyy_inv # B x A matrix
    Σxy_Σxx_inv = Σxy' * Σxx_inv # B x A matrix

    Σxy_Σyx_Σyy_inv = Σxy' * (Σyx' * Σyy_inv)

    # Compute derivatives
    dI_dx = -0.5 .* (Σxx_inv .* (Σxy_Σyx_Σyy_inv * Σyx_Σyy_inv') .+ Σxy_Σxx_inv) # T x A matrix
    dI_dy = -0.5 .* (Σyy_inv .* (Σyx_Σyy_inv * Σxy_Σyx_Σyy_inv) .+ Σyx_Σyy_inv) # T x B matrix

    return dI_dx, dI_dy
end
