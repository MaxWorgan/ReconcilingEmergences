using ReconcilingEmergences
using Test

@testset "ReconcilingEmergences.jl" begin
    @test ReconcilingEmergences.unique_rows([1,10,10,3,10]) == [1,2,2,3,2]
    @test sort(ReconcilingEmergences.index_frequencies([1,2,3,2,1])) == [0.2,0.4,0.4]
    @test ReconcilingEmergences.marginal_entropies([1,2,3,2,1], 1e-8) == -8.754887502163468
end
