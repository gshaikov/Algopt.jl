module TestUniDerivative

using Test
using Algopt.UniDerivative:
df,
CentralDiff, ComplexDiff

quadratic = x->x^2

@testset "df central" begin
    @test df(CentralDiff(), quadratic, 3) ≈ 2 * 3
end

@testset "df complex" begin
    @test df(ComplexDiff(), quadratic, 3) ≈ 2 * 3
end

end
