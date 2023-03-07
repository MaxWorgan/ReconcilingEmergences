using CUDA

function logdet!(m)
    X_d, _ = CUSOLVER.getrf!(m)
    ldet   = sum(log, diag(X_d))
    return ldet
end
 
LinearAlgebra.logdet(A::CuArray) = logdet!(copy(A))