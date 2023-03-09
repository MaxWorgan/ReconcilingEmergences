using CUDA, Folds, Transducers, FoldsCUDA

function EmergencePsi(X, V, tau=1, method="gaussian")
    ## EMERGENCEPSI Compute causal emergence criterion from data
    #
    #     PSI = EMERGENCEPSI[X, V] computes the causal emergence criterion psi for
    #     the system with micro time series X & macro time series Y. Micro data
    #     must be of size TxD & macro data of size TxR; where T is the length of
    #     the time series & D;R the dimensions of X;V respectively. Note that for
    #     the theory to hold V has to be a [possibly stochastic] function of X.
    #
    #     PSI = EMERGENCEPSI[X, V, TAU] uses a time delay of TAU samples to
    #     compute time-delayed mutual information. (default: 1)
    #
    #     PSI = EMERGENCEPSI[X, V, TAU, METHOD] estimates mutual info using a
    #     particular METHOD. Can be "discrete' | 'gaussian". If empty; it will
    #     try to infer the most suitable method.
    #
    #     [PSI, V_MI, X_MI] = EMERGENCEPSI[X, V, ...] also returns the mutual
    #     info in macro & micro variables; such that PSI = V_MI - X_MI.
    #
    # Reference:
    #     Rosas*, Mediano*, et al. (2020). Reconciling emergences: An
    #     information-theoretic approach to identify causal emergence in
    #     multivariate data. https://arxiv.org/abs/2004.08220
    #
    # Pedro Mediano & Fernando Rosas; Aug 2020

    if size(V, 1) ≠ size(X, 1)
        error("X & V must have the same height.")
    end

    if lowercase(method) == "gaussian"
        MI_fun = GaussianMI
    elseif lowercase(method) == "discrete"
        MI_fun = DiscreteMI
    else
        error("Unknown method")
    end

    ## Compute mutual infos & psi()
    v_mi = CUDA.@allowscalar @views GaussianMI(V[1:end-tau, :], V[1+tau:end, :])

    # x_mi = CUDA.@allowscalar mapfoldl(j -> GaussianMI(X[1:end-tau, j], V[1+tau:end, :]), add!, 1:size(X, 2); init=0.0)
    x_mi = CUDA.@allowscalar @views Folds.mapreduce(k -> GaussianMI(X[1:end-tau, k], V[1+tau:end, :]),+, 1:size(X,2))

    return v_mi - x_mi

end