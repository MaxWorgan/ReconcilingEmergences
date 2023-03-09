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
    mi  = 0.5 * (@views logdet(rho[1:Dx, 1:Dx]) + logdet(rho[Dx+1:end, Dx+1:end]) - logdet(rho))

    return mi

end

