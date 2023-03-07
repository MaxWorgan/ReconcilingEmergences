using StatsBase 

function DiscreteMI(X, Y, tolerance=1e-8)

    if (size(X, 1) ≠ size(Y, 1))
        error("X & Y must be matrices of the same height.")
    end

    sX = unique_rows(X)
    Hx = marginal_entropies(index_frequencies(sX), tolerance)

    sY = unique_rows(Y)
    Hy = marginal_entropies(index_frequencies(sY), tolerance)

    ## Make joint distribution; & estimate MI
    j = maximum(sY) * (sX .- 1) .+ sY
    Hxy = marginal_entropies(index_frequencies(j), tolerance)

    mi = Hx + Hy - Hxy

    return mi

end

## Compute marginal entropies
function marginal_entropies(p::AbstractArray{T}, tol) where {T<:Real}
    s = zero(T)
    for i = 1:length(p)
        @inbounds pi = p[i]
        if pi > tol
            s += pi * log2(pi)
        end
    end
    return -s
end

# create the 'ic' output from matlabs unique function
# @see https://uk.mathworks.com/help/matlab/ref/double.unique.html
# TODO: this doesn't work for CuArrays
function unique_rows(x::AbstractArray)
    cc = mapreduce(string, *, x, dims=2)
    return indexin(cc,sort(unique(cc)))
end

function index_frequencies(x)
    return collect(values(proportionmap(x)))
end
