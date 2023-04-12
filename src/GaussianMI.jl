using Statistics
using Distributions
using LinearAlgebra
using CUDA

function GaussianMI(X, Y)
    if (size(X, 1) ≠ size(Y, 1))
        error("X & Y must be matrices of the same height. $(axes(X)) $(axes(Y))")
    end

    Dx  = size(X, 2)
    rho = cor([X Y])

    ld1 = logdet(rho[1:Dx, 1:Dx])
    ld2 = logdet(rho[Dx+1:end, Dx+1:end])
    ld3 = logdet(rho)       

    mi  = 0.5 * (ld1 + ld2 - ld3)

    return mi

end

