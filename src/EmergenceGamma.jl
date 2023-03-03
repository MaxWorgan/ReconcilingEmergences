function EmergenceGamma(X, V, tau=1, method="gaussian")
    ## EMERGENCEGAMMA Compute causal decoupling criterion from data
    #
    #     GAMMA = EMERGENCEGAMMA[X, V] computes the causal decoupling criterion
    #     gamma for the system with micro time series X & macro time series Y.
    #     Micro data must be of size TxD & macro data of size TxR; where T is the
    #     length of the time series & D;R the dimensions of X;V respectively.
    #     Note that for the theory to hold V has to be a [possibly stochastic]
    #     function of X.
    #
    #     GAMMA = EMERGENCEGAMMA[X, V, TAU] uses a time delay of TAU samples to
    #     compute time-delayed mutual information. (default: 1)
    #
    #     GAMMA = EMERGENCEGAMMA[X, V, TAU, METHOD] estimates mutual info using a
    #     particular METHOD. Can be "discrete' | 'gaussian". If empty; it will
    #     try to infer the most suitable method.
    #
    # Reference:
    #     Rosas*, Mediano*, et al. (2020). Reconciling emergences: An
    #     information-theoretic approach to identify causal emergence in
    #     multivariate data. https://arxiv.org/abs/2004.08220
    #
    # Pedro Mediano & Fernando Rosas; Aug 2020
    
    ## Parameter checks & initialisation
    if ndims(X) ≠ 2 || ndims(V) ≠ 2
      error("X & V have to be 2D matrices.")
    end
    if size(V,1) ≠ size(X,1)
      error("X & V must have the same height.")
    end

    
    if lowercase(method) == "gaussian"
      MI_fun = GaussianMI
    elseif lowercase(method) ==  "discrete"
      MI_fun = DiscreteMI
    else
      error("Unknown method")
    end
    
    
    ## Compute mutual infos & gamma()
    gamma = maximum(map(j -> MI_fun(V[1:end-tau,:], X[1+tau:end,j]), 1:size(X,2)))

    return gamma
    
    
end