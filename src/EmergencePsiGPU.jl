using CUDA


function EmergencePsiGPU(x,v, tau=1)

    c_index = [CartesianIndices((1:(size(x,1) -tau), 1:(size(v,2) + tau), i:i)) for i in 1:size(x)[end]] |> cu
    logdet_rho = CUDA.zeros(Float32, size(x)[end])
    logdet_s2rho = CUDA.zeros(Float32, size(x)[end])

    C1 = CUDA.zeros(Float32, size(x,1)-1, size(v,2)+1, size(x)[end])
    rho1 = CUDA.zeros(Float32, size(C1,2), size(C1,2), size(x)[end])
    col_means1 = CUDA.zeros(size(x)[end], 11)
    col_std1 = CUDA.zeros(size(x)[end], 11)

    C2 = CUDA.zeros(Float32, size(v,1)-1, size(v,2)*2)
    rho2 = CUDA.zeros(Float32, size(C2,2), size(C2,2))
    col_means2 = CUDA.zeros(size(C2)[2])
    col_std2 = CUDA.zeros(size(C2)[2])

    x_mi = compute_reduced_MI(x[1:end-tau,:], v[1+tau:end,:], C1, rho1, c_index, col_means1, col_std1, logdet_rho, logdet_s2rho)

    v_mi = compute_single_MI(v[1:end-tau,:], v[1+tau:end,:], C2, col_means2, col_std2, rho2)

    return v_mi - x_mi


end
