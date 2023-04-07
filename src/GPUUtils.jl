using CUDA
using LinearAlgebra

function slogdet(m)
    # X_d, _ = CUSOLVER.getrf!(m)
    X_d    = cholesky(m, check=false)
    # ldet   = 2 * sum(log, diag(X_d.L))
    return logdet(X_d) 
end
 
Mylogdet(a) = slogdet(a)