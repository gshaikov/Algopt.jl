module UniDerivativeTests

import ..Algopt # ensure we are using the correct Algopt

using Test

using Algopt.UniDerivative:
df,
CentralDiff, ComplexDiff

quadratic = x -> x^2

@testset "derivative central method" begin
    @test df(CentralDiff(), quadratic, 3) ≈ 2 * 3
end

@testset "derivative complex method" begin
    @test df(ComplexDiff(), quadratic, 3) ≈ 2 * 3
end

end # module
