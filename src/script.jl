using CUDA
using Statistics
using LinearAlgebra

num_of_samples = 20000

x = CUDA.rand(Float32, 48, num_of_samples) |> cu
v = CUDA.rand(Float32, 48, 10) |> cu
c_index = [CartesianIndices((1:(size(x,1) -1), 1:(size(v,2) + 1), i:i)) for i in 1:size(x)[end]] |> cu
logdet_rho = CUDA.zeros(Float32, size(x)[end])
logdet_s2rho = CUDA.zeros(Float32, size(x)[end])

C = CUDA.zeros(Float32, 47, 11, size(x)[end])
rho1 = CUDA.zeros(Float32, 11, 11, size(x)[end])
rho2 = CUDA.zeros(Float32, 10, 10, size(x)[end])
rho3 = CUDA.zeros(Float32, 1, 1, size(x)[end])
col_means = CUDA.zeros(size(x)[end], 11)
col_std = CUDA.zeros(size(x)[end], 11)

@cuda threads=384 blocks=80 calculate_cor_kernel_batched(x[1:end-1,:], v[2:end,:], C, c_index, col_means, col_std, rho1, rho2)

rho2
reshaped_rho  = CuArray(map(pointer, eachslice(rho1, dims=3)))
reshaped_rho2 = CuArray(map(pointer, eachslice(rho2, dims=3)))

output = callBatchedCholesky(reshaped_rho, 11)



compute_reduced_MI(x[1:end-1,:],v[2:end, :], C, rho1,  rho2, c_index, col_means, col_std, logdet_rho, logdet_s2rho)

C = CUDA.zeros(Float32, 47, size(v,2) + size(v,2))
rho = CUDA.zeros(Float32, size(C,2), size(C,2))
col_means = CUDA.zeros(size(C,2))
col_std = CUDA.zeros(size(C,2))
@cuda threads=512 blocks=80 calculate_cor_kernel(v[1:end-1,:],v[2:end,:],C,col_means, col_std,rho)
rho

compute_single_MI(v[1:end-1,:],v[2:end, :], C, col_means, col_std, rho)
rho


EmergencePsiGPU(x,v)


C

GaussianMI(v[1:end-1,:],v[2:end,:])


t = @cuda launch=fa lse logdet_kernel(rho,logdet_rho)

launch_configuration(t.fun)

s = 0.0f32
for i in 1:size(x,2)
    s += GaussianMI(x[1:end-1,i],v[2:end,:])
end
s

s

using BenchmarkTools
logdet_rho
logdet_s2rho

col_std
col_means

rho

C

CUSOLVER.potrfBatched!