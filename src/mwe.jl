using CUDA
using LinearAlgebra
using Statistics
using Distributions
using Folds
using FoldsCUDA
using GPUArrays

function mylogdet(V::CuArray{Float32})
    CH = cholesky(V; check=false)
    return  2.0 * sum(log, diag(CH.U))
end

function calc_mi(X, Y, Dx=1)
    rho   = Statistics.corzm(hcat(X,Y))
    return 0.5 * (@views mylogdet(rho[1:Dx, 1:Dx]) + mylogdet(rho[Dx+1:end, Dx+1:end]) - mylogdet(rho))
end

function calc_mi(X, Y, k, Dx=1)
    input::CuArray{Float32} = Statistics.covzm(X)
    # rho = cov(hcat(X[1:end-tau,k],Y[1+tau:end,:]))
    return 0.5# * (@views mylogdet(rho[1:Dx, 1:Dx]) + mylogdet(rho[Dx+1:end, Dx+1:end]) - mylogdet(rho))
end

function calc_ψ(X,Y,tau=1)
   return Folds.mapreduce(k -> calc_mi(X,Y,k),+, 1:size(X,2), CUDAEx())
end


X = CuArray(rand(Float32,10,10))

Y :: CuArray{Float32} = CuArray(4*rand(Float32,10,5).-2)


X2 :: CuArray{Float32} = CuArray(rand(Float32,10,10))
Y2 :: CuArray{Float32} = CuArray(4*rand(Float32,10,5).-2)


@btime calc_mi(x[1:end-1,1],v[2:end,:])
@time calc_ψ(x,v)

calc_ψ(X,Y)

Y[:,4]


y3 = @view(Y2[2:end,2:end])
using FLoops

Y

# try 
    @floop CUDAEx() for z in eachcol(X)
      #  CUDA.@cuprintln(z)
#        CUDA.@cuprintln(y3)
        hcat(z,y3)
        # i = a
        # rho = Statistics.covzm(i)
        # Dx = size(rho,2)
        # @allowscalar test = @view(rho[2:Dx,2:end])
        # @reduce tot += 0.5 * (log(det(rho[1, 1])) + log(det(rho[2:end, 2:end])) - log(det((rho))))
        # @reduce tot += 0.5 * @views log(det(rho[1,1])) # + log(det(rho[Val(2):Val(end),Val(2):end]))
        @reduce (tot += z[1,1])
    end

# catch err
#     code_typed(err)
# entot

tot


Threads.nthreads()

x[1:end-1, 1]

first(eachcol(x))

using Tullio, CUDAKernels,KernelAbstractionso

mult(A,B) = @tullio T[a,b] := hcat(A[a,c],B[a,b], dims=2) ∘ cor


mult(x,v)



@device_code_warntype interactive=true @tullio Z[a,b] := calc_mi(x[a,z], v[a,b])

hcat(x[:,1],v)
@tullio S[1,c] := M[r,c]  # sum over r ∈ 1:3, for each c ∈ 1:7


M = rand(1:20, 3, 7)
@tullio S[1,c] := M[r,c]  # sum over r ∈ 1:3, for each c ∈ 1:7
S